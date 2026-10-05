//! Prompt-conditioned prose fit on an immutable River expert snapshot.
//!
//! Loads one expert snapshot read-only (`load_universal_checkpoint`), applies
//! in-memory overrides (layer rates, relax steps, eta, byte-head scale, byte
//! target encoding, SEAL), and trains it on `datasets/fits/prose-fit-v1.jsonl`
//! with the production construction for prepared responses:
//!
//! * records go through the production tagged-record loader
//!   (`load_tagged_task_dataset`, `generation_prompt` context, every response
//!   position plus EOS, persistent-memory targets) and every row's input is
//!   asserted equal to the runtime encoding of
//!   `generation_prompt(decode_runtime_prompt(inputs), instructions)`;
//! * `inherited-byte`: `prepared_response_byte_example` -> `make_masked_batch`
//!   -> byte-head batch scale (`rows / reference`) -> `lift_multimodal_batch`
//!   -> `train_masked_batch_gpu_inherited_paths_with_seal`;
//! * `request-token`: `task_batches` ->
//!   `train_masked_batch_gpu_request_paths_with_seal` (trunk at `base_eta`).
//!
//! Training runs on the burn NdArray backend (the production functions, CPU).
//! Evaluation decodes every prompt with the shared greedy
//! `generate_text_with_scorer` against one batched settle that mirrors the
//! runtime scorers (`InheritedByteScorer` on CPU, `UniversalGpuScorer` on
//! device): inherited text re-initializes every layer bottom-up from each
//! context, the request-token path keeps a warm-started persistent state. One
//! worker thread per prompt runs the real generator while one batched settle
//! serves every live prompt.
//! Retention (before/after, same run config): 128 fixed Dracula windows (top-1
//! next byte on the trained path), the fixed Open-Jev typed held-out records
//! (Choice/Score/Noul, promotion grouping and pass rule), and, for the request
//! path, the fixed Dolly held-out sequence rows (promotion token gate).
//!
//! Nothing is written back to any checkpoint. Fitting 64 items says nothing
//! about generalization.
//!
//! ```text
//! cargo run --release --no-default-features --example prose_fit -- --help
//! ```

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::Instant,
};

use burn::backend::{ndarray::NdArrayDevice, NdArray};
use clap::{Parser, ValueEnum};
use ndarray::{Array2, ArrayView1, Axis, Slice};
use pcn::{
    byte_scoring_input, decode_inherited_input, decode_noul_state, decode_runtime_prompt,
    encode_sequence_input, generate_text_with_scorer, generation_prompt,
    gpu::{
        predict_batch_gpu, train_masked_batch_gpu_inherited_paths_with_seal,
        train_masked_batch_gpu_request_paths_with_seal, GpuInferenceSession, GpuPcn,
        MaskedEnergyGuard, SessionStart,
    },
    lift_inherited_input, lift_multimodal_batch, load_tagged_task_dataset, load_universal_checkpoint,
    make_masked_batch, prepared_response_byte_example, task_batches, BatchState, ByteScoreProvider,
    ByteTargetEncoding, GenerationError, MaskedBatch, MaskedPcnConfig, Modality, OutputMode,
    SealConfig, SurpriseState, TaskSupervision, TaskTrainingExample, BYTE_CONTEXT_BYTES,
    BYTE_EOS_INDEX, BYTE_OUTPUT_OFFSET, BYTE_SUPPORT_DIM, CONDITION_INPUT_ROWS, GENERIC_NOUL_INDEX,
    MULTIMODAL_INPUT_DIM, PCN, TASK_HOLDOUT_DIVISOR, TOKEN_SUPPORT_START, UNIVERSAL_INPUT_DIM,
};
use serde::Deserialize;
use serde_json::{json, Value};

type Cpu = NdArray<f32>;
type Error = Box<dyn std::error::Error>;

/// Public fixture `max_bytes` for prose and code text answers.
const PROSE_MAX_BYTES: usize = 32;
const CODE_MAX_BYTES: usize = 48;
const KINDS: [&str; 3] = ["prose", "multi-turn", "code"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputPath {
    /// Visible today: the inherited expert's byte head (unpromoted requests).
    InheritedByte,
    /// The request expert's token support (used once sequence promotion passes).
    RequestToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum UpdateScope {
    Inherited,
    Request,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Decoder {
    /// Native CPU settle (`GenerationSession` / CPU runtime scorer arithmetic).
    Native,
    /// burn `GpuInferenceSession` on NdArray (the live device scorer's code).
    Burn,
}

#[derive(Debug, Parser)]
struct Args {
    /// Expert snapshot directory (read-only), e.g. `<set>/<generation>/request-conditioned`.
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, value_enum)]
    path: OutputPath,
    /// Expert serving typed outputs when `--path inherited-byte` (evaluated once,
    /// untouched by such runs). Defaults to the sibling `request-conditioned`.
    #[arg(long)]
    typed_checkpoint: Option<PathBuf>,
    #[arg(long, default_value = "datasets/fits/prose-fit-v1.jsonl")]
    fit: PathBuf,
    /// Training epochs over the fit set; 0 evaluates only.
    #[arg(long, default_value_t = 0)]
    epochs: usize,
    #[arg(long, default_value_t = 1)]
    eval_every: usize,
    /// Byte cap for evaluations before the final one (13 decides exact match for
    /// <=12-byte answers). The final evaluation always uses the fixture caps.
    #[arg(long)]
    eval_max_bytes: Option<usize>,
    #[arg(long, value_enum, default_value_t = Decoder::Native)]
    decoder: Decoder,
    /// Backend for the cold retention predictions (Dracula, held-out typed/sequence).
    #[arg(long, value_enum, default_value_t = Decoder::Native)]
    retention_backend: Decoder,
    /// Expert being trained: `request` uses the request-path update with the trunk
    /// at `--base-eta` and the request profile, and its typed/sequence retention is
    /// measured on the trained expert. Default follows `--path`.
    #[arg(long, value_enum)]
    update_scope: Option<UpdateScope>,
    /// DIAGNOSTIC, in memory: zero the top weight columns `a..b` before training
    /// (as overfit_bytes `--silence-top-columns`). Broad weight surgery; never production.
    #[arg(long)]
    silence_top_columns: Option<String>,
    /// Override the checkpoint relaxation steps.
    #[arg(long)]
    relax_steps: Option<usize>,
    #[arg(long)]
    alpha: Option<f32>,
    /// Comma-separated non-input layer rates; default is the role's profile.
    #[arg(long)]
    layer_alphas: Option<String>,
    /// Default: production role eta (inherited `--inherited-eta` 1e-7, request
    /// checkpoint `masked_pcn.eta`).
    #[arg(long)]
    eta: Option<f32>,
    /// Request-path trunk eta (production `--inherited-eta`).
    #[arg(long, default_value_t = 1.0e-7)]
    base_eta: f32,
    /// Production `--byte-head-reference-batch-size`.
    #[arg(long, default_value_t = 64)]
    byte_head_reference_batch_size: usize,
    /// Extra multiplier on the inherited byte-head update scale (production 1).
    #[arg(long, default_value_t = 1.0)]
    byte_head_extra: f32,
    /// Production `--corpus-batch-size` (inherited prepared-response rows).
    #[arg(long, default_value_t = 1024)]
    corpus_batch_size: usize,
    /// Production `--task-batch-size` (request rows).
    #[arg(long, default_value_t = 64)]
    task_batch_size: usize,
    /// `signed` or `zero`; default is the checkpoint's committed encoding.
    #[arg(long)]
    byte_target_encoding: Option<ByteTargetEncoding>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    seal: bool,
    #[arg(long, default_value_t = 10_000_000.0)]
    max_energy: f32,
    /// Live service value.
    #[arg(long, default_value_t = 160)]
    max_relax_steps: usize,
    #[arg(long, default_value = "/bulk-storage/datasets/books/dracula.txt")]
    retention_text: PathBuf,
    #[arg(long, default_value_t = 128)]
    retention_windows: usize,
    #[arg(long, default_value = "/fast-storage/river-datasets/zefancai-open-jev-release-v2-v1")]
    typed_dataset: PathBuf,
    #[arg(long, default_value = "zefancai-open-jev-release-v2-v1")]
    typed_dataset_id: String,
    #[arg(long, default_value = "/fast-storage/river-datasets/databricks-dolly-15k-v1")]
    sequence_dataset: PathBuf,
    #[arg(long, default_value = "databricks-dolly-15k-v1")]
    sequence_dataset_id: String,
    /// Retention gate: maximum allowed absolute drop (Dracula top-1, typed rank accuracy).
    #[arg(long, default_value_t = 0.02)]
    max_retention_drop: f64,
    /// Retention gate: maximum allowed typed grouped-Brier rise.
    #[arg(long, default_value_t = 0.01)]
    max_brier_rise: f64,
    /// Stop after this many consecutive non-finite/guard-skipped batches.
    #[arg(long, default_value_t = 3)]
    stop_consecutive_skips: usize,
    /// Stop at the first evaluation at or after this epoch where the trained
    /// items do not beat the epoch-0 baseline in exact answers or distinct outputs.
    #[arg(long, default_value_t = 4)]
    stop_patience_epochs: usize,
    /// Comma-separated fit item ids to train on (default: all). Intermediate
    /// evaluations decode only these; the final evaluation decodes every item.
    #[arg(long)]
    train_ids: Option<String>,
    /// Training budget in minutes: the training loop and its intermediate
    /// evaluations. Checked before each epoch and each batch; no exemptions.
    #[arg(long, default_value_t = 40.0)]
    max_minutes: f64,
    /// Budget in minutes for the epoch-0 baseline evaluation (fails closed).
    #[arg(long, default_value_t = 15.0)]
    max_baseline_minutes: f64,
    /// Budget in minutes for the final fixture-cap evaluation.
    #[arg(long, default_value_t = 30.0)]
    max_final_minutes: f64,
    #[arg(long, default_value_t = 0x5052_4f53_45)]
    seed: u64,
    #[arg(long, default_value = "/tmp/prosefit")]
    out_dir: PathBuf,
    #[arg(long)]
    label: String,
}

#[derive(Debug, Clone, Deserialize)]
struct FitItem {
    id: String,
    kind: String,
    prompt: String,
    instructions: String,
    answer: String,
    fact: Option<String>,
    /// Conversation pair: same final question, different earlier fact and answer.
    pair: Option<String>,
}

impl FitItem {
    fn request_inputs(&self) -> Value {
        if self.kind == "code" {
            json!({"prompt": self.prompt, "modality": "code"})
        } else {
            json!({"prompt": self.prompt})
        }
    }

    fn record_kind(&self) -> &'static str {
        if self.kind == "code" {
            "code-instruction"
        } else {
            "instruction-response"
        }
    }

    fn max_bytes(&self) -> usize {
        if self.kind == "code" {
            CODE_MAX_BYTES
        } else {
            PROSE_MAX_BYTES
        }
    }
}

struct Logger {
    file: fs::File,
    label: String,
}

impl Logger {
    fn log(&mut self, kind: &str, mut record: Value) -> Result<(), Error> {
        record["label"] = json!(self.label);
        record["kind"] = json!(kind);
        println!("{record}");
        writeln!(self.file, "{record}")?;
        self.file.flush()?;
        Ok(())
    }
}

fn load_fit_set(path: &Path) -> Result<Vec<FitItem>, Error> {
    let items: Vec<FitItem> = fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let answers: BTreeSet<&str> = items.iter().map(|item| item.answer.as_str()).collect();
    let ids: BTreeSet<&str> = items.iter().map(|item| item.id.as_str()).collect();
    if answers.len() != items.len() || ids.len() != items.len() {
        return Err("fit set answers and ids must be distinct".into());
    }
    for item in &items {
        if !KINDS.contains(&item.kind.as_str()) {
            return Err(format!("{}: unknown kind {}", item.id, item.kind).into());
        }
        if !(3..=12).contains(&item.answer.len())
            || !item.answer.bytes().all(|byte| (b' '..=b'~').contains(&byte))
        {
            return Err(format!("{}: answer must be 3-12 printable ASCII bytes", item.id).into());
        }
        // The tagged loader trims fields and splits on tags.
        for text in [&item.prompt, &item.instructions, &item.answer] {
            if text.contains('<') || text.trim() != text.as_str() || text.is_empty() {
                return Err(format!("{}: fields must be trimmed, nonempty, tag-free", item.id).into());
            }
        }
        if item.kind == "multi-turn" {
            let fact = item.fact.as_deref().ok_or("multi-turn item missing fact")?.to_lowercase();
            let (prompt, _) = decode_runtime_prompt(&item.request_inputs())?;
            let context = String::from_utf8(generation_prompt(&prompt, &item.instructions))?;
            let split = context.len() - BYTE_CONTEXT_BYTES;
            if context[split..].to_lowercase().contains(&fact)
                || !context[..split].to_lowercase().contains(&fact)
            {
                return Err(format!("{}: fact must lie only before the last 64 context bytes", item.id).into());
            }
        }
    }
    let mut pairs = BTreeMap::<&str, Vec<&FitItem>>::new();
    for item in &items {
        match (item.kind.as_str(), item.pair.as_deref()) {
            ("multi-turn", Some(pair)) => pairs.entry(pair).or_default().push(item),
            ("multi-turn", None) => return Err(format!("{}: multi-turn item missing pair", item.id).into()),
            (_, Some(_)) => return Err(format!("{}: only multi-turn items are paired", item.id).into()),
            _ => {}
        }
    }
    for (pair, members) in &pairs {
        let tails: BTreeSet<&str> = members.iter().map(|item| item.prompt.rsplit('\n').next().unwrap_or("")).collect();
        // Identical final question, different fact: a question->answer mapping cannot pass both.
        if members.len() != 2 || tails.len() != 1 || members[0].fact == members[1].fact {
            return Err(format!("{pair}: needs two items with one final question and different facts").into());
        }
    }
    Ok(items)
}

/// Production sequence rows for every fit item via the tagged-record loader.
/// Holdout ordinals (multiples of `TASK_HOLDOUT_DIVISOR`) get unused filler
/// records so that every fit item is a training record.
fn fit_training_examples(
    items: &[FitItem],
    dir: &Path,
    seed: u64,
) -> Result<Vec<Vec<TaskTrainingExample>>, Error> {
    fs::create_dir_all(dir)?;
    let mut text = String::new();
    let mut ordinal = 0u64;
    for item in items {
        while ordinal % TASK_HOLDOUT_DIVISOR == 0 {
            text.push_str("<river-example kind=\"instruction-response\"><instruction>Unused holdout slot.</instruction><response>unused</response></river-example>\n");
            ordinal += 1;
        }
        text.push_str(&format!(
            "<river-example kind=\"{}\"><instruction>{}</instruction><context>{}</context><response>{}</response></river-example>\n",
            item.record_kind(), item.instructions, item.prompt, item.answer,
        ));
        ordinal += 1;
    }
    fs::write(dir.join("prose-fit.txt"), text)?;
    let load = load_tagged_task_dataset(dir, "prose-fit-v1", "instruction-response", 0, usize::MAX, seed)?;
    if load.selected_records != items.len() {
        return Err("fit loader did not select every fit record".into());
    }
    let mut rows = load.training.into_iter();
    let mut grouped = Vec::with_capacity(items.len());
    for item in items {
        let (prompt, modality) = decode_runtime_prompt(&item.request_inputs())?;
        let mut context = generation_prompt(&prompt, &item.instructions);
        let answer = item.answer.as_bytes();
        let mut examples = Vec::with_capacity(answer.len() + 1);
        for position in 0..=answer.len() {
            let example = rows.next().ok_or("fit loader returned too few rows")?;
            let expected_token = answer.get(position).map_or(BYTE_EOS_INDEX, |byte| usize::from(*byte));
            let TaskSupervision::Token { token, .. } = &example.supervision else {
                return Err("fit row is not token supervision".into());
            };
            // Parity with the runtime: the same request inputs and instructions
            // produce exactly the trained context at every response position.
            if *token != expected_token
                || example.input != encode_sequence_input(&context, modality, OutputMode::Text)?
            {
                return Err(format!("{}: training row {position} differs from the runtime context", item.id).into());
            }
            if position < answer.len() {
                context.push(answer[position]);
            }
            examples.push(example);
        }
        grouped.push(examples);
    }
    if rows.next().is_some() {
        return Err("fit loader returned extra rows".into());
    }
    Ok(grouped)
}

/// The runtime scorers' per-step input for a growing context.
fn step_input(token_path: bool, context: &[u8], modality: Modality) -> Result<Vec<f32>, Error> {
    Ok(byte_scoring_input(context, modality, OutputMode::Text, token_path)?.to_vec())
}

fn rows_matrix(rows: &[Vec<f32>]) -> Result<Array2<f32>, Error> {
    let width = rows.first().map_or(UNIVERSAL_INPUT_DIM, Vec::len);
    Ok(Array2::from_shape_vec((rows.len(), width), rows.concat())?)
}

/// Settle shared by all live prompts. Rows are independent, so batching does not
/// change any row's dynamics.
trait Settler {
    fn settle(&mut self, rows: &[usize], inputs: &Array2<f32>) -> Result<Array2<f32>, Error>;
}

struct NativeSettler<'a> {
    pcn: &'a PCN,
    config: &'a MaskedPcnConfig,
    state: BatchState,
    rows: Vec<usize>,
    /// Re-initialize from each input (inherited text) instead of carrying state.
    fresh: bool,
}

/// `init_state_from_input`: x[l] = tanh(W[l]^T x[l-1]) per row.
fn init_layers_from_input(pcn: &PCN, state: &mut BatchState) {
    for layer in 1..pcn.dims().len() {
        state.x[layer] = state.x[layer - 1].dot(&pcn.w[layer]).mapv(f32::tanh);
    }
}

impl<'a> NativeSettler<'a> {
    fn new(
        pcn: &'a PCN,
        config: &'a MaskedPcnConfig,
        initial: &Array2<f32>,
        fresh: bool,
    ) -> Result<Self, Error> {
        let mut state = pcn.init_batch_state(initial.nrows());
        state.x[0].assign(initial);
        init_layers_from_input(pcn, &mut state);
        if !fresh {
            pcn.relax_batch(&mut state, config.relax_steps, config.alpha, &config.layer_alphas)?;
        }
        Ok(Self { pcn, config, state, rows: (0..initial.nrows()).collect(), fresh })
    }
}

impl Settler for NativeSettler<'_> {
    fn settle(&mut self, rows: &[usize], inputs: &Array2<f32>) -> Result<Array2<f32>, Error> {
        if rows != self.rows.as_slice() {
            let keep: Vec<usize> = rows
                .iter()
                .map(|row| self.rows.iter().position(|kept| kept == row).ok_or("finished row revived"))
                .collect::<Result<_, _>>()?;
            for layer in self.state.x.iter_mut().chain(&mut self.state.mu).chain(&mut self.state.eps) {
                *layer = layer.select(Axis(0), &keep);
            }
            self.state.batch_size = rows.len();
            self.rows = rows.to_vec();
        }
        self.state.x[0].assign(inputs);
        if self.fresh {
            init_layers_from_input(self.pcn, &mut self.state);
        }
        self.pcn.relax_batch(
            &mut self.state, self.config.relax_steps, self.config.alpha, &self.config.layer_alphas,
        )?;
        Ok(self.state.x[self.state.x.len() - 1].clone())
    }
}

struct BurnSettler<'a> {
    session: GpuInferenceSession<'a, Cpu>,
    last: Array2<f32>,
}

impl Settler for BurnSettler<'_> {
    fn settle(&mut self, rows: &[usize], inputs: &Array2<f32>) -> Result<Array2<f32>, Error> {
        for (index, row) in rows.iter().enumerate() {
            self.last.row_mut(*row).assign(&inputs.row(index));
        }
        let output = self.session.settle(&self.last)?;
        Ok(output.select(Axis(0), rows))
    }
}

enum Message {
    Scores(usize, Vec<u8>),
    Done(usize, Result<String, GenerationError>),
}

/// Routes one prompt's `generate_text_with_scorer` calls to the batched settle.
struct RowScorer {
    row: usize,
    requests: mpsc::Sender<Message>,
    replies: mpsc::Receiver<[f32; 257]>,
}

impl ByteScoreProvider for RowScorer {
    // Text generation never snapshots; JSON decoding is not used here.
    type Snapshot = ();

    fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
        self.requests
            .send(Message::Scores(self.row, context.to_vec()))
            .map_err(|_| GenerationError::InvalidConfig)?;
        self.replies.recv().map_err(|_| GenerationError::InvalidConfig)
    }

    fn snapshot(&self) -> Self::Snapshot {}

    fn restore(&mut self, (): Self::Snapshot) {}
}

struct Generated {
    text: Result<String, String>,
    first_scores: [f32; 257],
}

struct GenerationRequest {
    prompt: Vec<u8>,
    modality: Modality,
    max_bytes: usize,
    initial: Vec<f32>,
}

fn generation_requests(items: &[FitItem], cap: Option<usize>) -> Result<Vec<GenerationRequest>, Error> {
    items
        .iter()
        .map(|item| {
            let inputs = item.request_inputs();
            let (prompt, modality) = decode_runtime_prompt(&inputs)?;
            Ok(GenerationRequest {
                prompt: generation_prompt(&prompt, &item.instructions),
                modality,
                max_bytes: cap.map_or(item.max_bytes(), |cap| cap.min(item.max_bytes())),
                initial: lift_inherited_input(&decode_inherited_input(&inputs)?).to_vec(),
            })
        })
        .collect()
}

/// Run the shared greedy generator for every request concurrently: one worker
/// thread per prompt, one batched settle per generated byte position.
fn generate_all(
    settler: &mut dyn Settler,
    requests: &[GenerationRequest],
    token_path: bool,
    deadline: Option<Instant>,
) -> Result<Option<Vec<Generated>>, Error> {
    let start = if token_path { TOKEN_SUPPORT_START } else { BYTE_OUTPUT_OFFSET };
    let mut texts: Vec<Option<Result<String, String>>> = (0..requests.len()).map(|_| None).collect();
    let mut first: Vec<Option<[f32; 257]>> = vec![None; requests.len()];
    let mut timed_out = false;
    thread::scope(|scope| -> Result<(), Error> {
        let (sender, receiver) = mpsc::channel();
        let mut replies = Vec::with_capacity(requests.len());
        for (row, request) in requests.iter().enumerate() {
            let (reply_sender, reply_receiver) = mpsc::channel();
            replies.push(reply_sender);
            let sender = sender.clone();
            scope.spawn(move || {
                let mut scorer = RowScorer { row, requests: sender.clone(), replies: reply_receiver };
                let result = generate_text_with_scorer(&mut scorer, &request.prompt, request.max_bytes);
                // The coordinator only stops listening after an error of its own.
                let _ = sender.send(Message::Done(row, result));
            });
        }
        drop(sender);
        let mut active = requests.len();
        let started = Instant::now();
        let mut round = 0usize;
        loop {
            let mut pending = BTreeMap::new();
            while pending.len() < active {
                match receiver.recv()? {
                    Message::Scores(row, context) => {
                        pending.insert(row, context);
                    }
                    Message::Done(row, result) => {
                        texts[row] = Some(result.map_err(|error| error.to_string()));
                        active -= 1;
                    }
                }
            }
            if pending.is_empty() {
                return Ok(());
            }
            if deadline.is_some_and(|deadline| Instant::now() > deadline) {
                // Dropping the reply channels ends every worker's generator.
                timed_out = true;
                return Ok(());
            }
            let rows: Vec<usize> = pending.keys().copied().collect();
            let inputs = rows_matrix(
                &pending
                    .iter()
                    .map(|(row, context)| step_input(token_path, context, requests[*row].modality))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            let output = settler.settle(&rows, &inputs)?;
            round += 1;
            eprintln!("generation round {round}: {} live rows, {:.0}s", rows.len(), started.elapsed().as_secs_f64());
            for (index, row) in rows.iter().enumerate() {
                let mut scores = [0.0; 257];
                for (offset, score) in scores.iter_mut().enumerate() {
                    *score = output[(index, start + offset)];
                }
                first[*row].get_or_insert(scores);
                replies[*row].send(scores)?;
            }
        }
    })?;
    if timed_out {
        return Ok(None);
    }
    texts
        .into_iter()
        .zip(first)
        .map(|(text, first)| {
            Ok(Generated {
                text: text.ok_or("generator did not finish")?,
                first_scores: first.ok_or("generator never scored")?,
            })
        })
        .collect::<Result<_, Error>>()
        .map(Some)
}

fn normalized(text: &str) -> String {
    let collapsed = text.trim().to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.strip_suffix('.').map_or(collapsed.clone(), str::to_owned)
}

fn repeated_four_gram(text: &[u8]) -> bool {
    let mut seen = BTreeSet::new();
    text.windows(4).any(|gram| !seen.insert(gram))
}

fn generation_report(items: &[FitItem], generated: &[Generated]) -> Value {
    let summarize = |selected: &[usize]| {
        let texts: Vec<&str> = selected
            .iter()
            .map(|index| generated[*index].text.as_deref().unwrap_or("<error>"))
            .collect();
        let exact = selected.iter().zip(&texts).filter(|(index, text)| **text == items[**index].answer).count();
        let normal = selected
            .iter()
            .zip(&texts)
            .filter(|(index, text)| normalized(text) == normalized(&items[**index].answer))
            .count();
        let distinct: BTreeSet<&str> = texts.iter().copied().collect();
        let mut counts = BTreeMap::<&str, usize>::new();
        for text in &texts {
            *counts.entry(text).or_default() += 1;
        }
        let mode = counts.values().copied().max().unwrap_or(0);
        let ranks: Vec<usize> = selected
            .iter()
            .map(|index| {
                let scores = &generated[*index].first_scores;
                let target = scores[usize::from(items[*index].answer.as_bytes()[0])];
                scores.iter().filter(|score| **score > target).count()
            })
            .collect();
        let repeated = texts.iter().filter(|text| repeated_four_gram(text.as_bytes())).count();
        let count = selected.len().max(1) as f64;
        json!({
            "items": selected.len(),
            "exact": exact, "exact_acc": exact as f64 / count,
            "normalized": normal, "normalized_acc": normal as f64 / count,
            "distinct_outputs": distinct.len(), "mode_output_count": mode,
            "mean_first_byte_rank": ranks.iter().sum::<usize>() as f64 / count,
            "first_byte_top1": ranks.iter().filter(|rank| **rank == 0).count(),
            "repeated_4gram_fraction": repeated as f64 / count,
            "errors": selected.iter().filter(|index| generated[**index].text.is_err()).count(),
        })
    };
    let all: Vec<usize> = (0..items.len()).collect();
    let mut by_kind = serde_json::Map::new();
    for kind in KINDS {
        let selected: Vec<usize> = all.iter().copied().filter(|index| items[*index].kind == kind).collect();
        by_kind.insert(kind.to_owned(), summarize(&selected));
    }
    // A pair passes only when both conversations are exactly right.
    let mut pairs = BTreeMap::<&str, (usize, usize)>::new();
    for (index, item) in items.iter().enumerate() {
        if let Some(pair) = item.pair.as_deref() {
            let entry = pairs.entry(pair).or_default();
            entry.0 += 1;
            entry.1 += usize::from(generated[index].text.as_deref() == Ok(item.answer.as_str()));
        }
    }
    let pair_correct = pairs.values().filter(|(members, correct)| members == correct).count();
    json!({
        "overall": summarize(&all),
        "by_kind": by_kind,
        "conversation_pairs": {"pairs": pairs.len(), "both_exact": pair_correct,
            "pair_acc": pair_correct as f64 / pairs.len().max(1) as f64},
        "outputs": items.iter().zip(generated).map(|(item, generated)| json!({
            "id": item.id, "answer": item.answer,
            "output": generated.text.as_ref().map_or_else(|error| format!("<error: {error}>"), Clone::clone),
        })).collect::<Vec<_>>(),
    })
}

fn evaluate_generation(
    cpu: &PCN,
    gpu: &GpuPcn<Cpu>,
    config: &MaskedPcnConfig,
    decoder: Decoder,
    items: &[FitItem],
    token_path: bool,
    cap: Option<usize>,
    deadline: Option<Instant>,
) -> Result<Option<Value>, Error> {
    let requests = generation_requests(items, cap)?;
    let initial = rows_matrix(&requests.iter().map(|request| request.initial.clone()).collect::<Vec<_>>())?;
    let generated = match decoder {
        Decoder::Native => {
            let mut settler = NativeSettler::new(cpu, config, &initial, !token_path)?;
            generate_all(&mut settler, &requests, token_path, deadline)?
        }
        Decoder::Burn => {
            let mut settler = BurnSettler {
                session: GpuInferenceSession::new(
                    gpu, &initial, config.relax_steps, config.alpha, &config.layer_alphas,
                    if token_path { SessionStart::Carry } else { SessionStart::FreshFromInput },
                    None,
                )?,
                last: initial.clone(),
            };
            generate_all(&mut settler, &requests, token_path, deadline)?
        }
    };
    Ok(generated.map(|generated| {
        let mut report = generation_report(items, &generated);
        report["max_bytes_cap"] = json!(cap);
        report
    }))
}

/// Settled model used for the cold (non-persistent) retention predictions.
#[derive(Clone, Copy)]
enum Model<'a> {
    /// Native CPU settle, as the CPU runtime's typed predictor.
    Native(&'a PCN),
    /// `predict_batch_gpu`, the promotion evaluator's predictor.
    Burn(&'a GpuPcn<Cpu>),
}

impl<'a> Model<'a> {
    const fn of(backend: Decoder, cpu: &'a PCN, gpu: &'a GpuPcn<Cpu>) -> Self {
        match backend {
            Decoder::Native => Self::Native(cpu),
            Decoder::Burn => Self::Burn(gpu),
        }
    }
}

/// Cold batched settle, 64 rows at a time (promotion evaluator chunking).
fn predict_rows(model: Model<'_>, config: &MaskedPcnConfig, rows: &[Vec<f32>]) -> Result<Array2<f32>, Error> {
    let mut outputs = Vec::with_capacity(rows.len());
    for chunk in rows.chunks(64) {
        let input = rows_matrix(chunk)?;
        let output = match model {
            Model::Native(pcn) => {
                // Carry mode settles the inputs at construction: one cold settle.
                let settled = NativeSettler::new(pcn, config, &input, false)?;
                settled.state.x[settled.state.x.len() - 1].clone()
            }
            Model::Burn(gpu) => {
                predict_batch_gpu(gpu, &input, config.relax_steps, config.alpha, &config.layer_alphas)
            }
        };
        outputs.extend(output.outer_iter().map(|row| row.to_vec()));
    }
    rows_matrix(&outputs)
}

fn argmax257(output: ArrayView1<'_, f32>, start: usize) -> usize {
    (0..BYTE_SUPPORT_DIM)
        .max_by(|left, right| output[start + *left].total_cmp(&output[start + *right]))
        .unwrap_or_default()
}

/// Fixed raw prose windows: 64 preceding bytes and the next byte.
fn retention_windows(path: &Path, count: usize) -> Result<Vec<(Vec<u8>, usize)>, Error> {
    let raw = fs::read(path)?;
    let find = |needle: &[u8]| raw.windows(needle.len()).position(|window| window == needle);
    let start = find(b"\nCHAPTER I\r\n").ok_or("retention start marker missing")? + 1;
    let end = find(b"*** END OF THE PROJECT").unwrap_or(raw.len());
    let span = end - start - BYTE_CONTEXT_BYTES - 1;
    Ok((0..count)
        .map(|index| {
            let position = start + BYTE_CONTEXT_BYTES + index * span / count;
            (raw[position - BYTE_CONTEXT_BYTES..position].to_vec(), usize::from(raw[position]))
        })
        .collect())
}

fn prose_retention(
    model: Model<'_>,
    config: &MaskedPcnConfig,
    windows: &[(Vec<u8>, usize)],
    token_path: bool,
) -> Result<Value, Error> {
    let rows = windows
        .iter()
        .map(|(context, _)| step_input(token_path, context, Modality::Prose))
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = predict_rows(model, config, &rows)?;
    let start = if token_path { TOKEN_SUPPORT_START } else { BYTE_OUTPUT_OFFSET };
    let mut predicted = BTreeMap::<usize, usize>::new();
    let mut non_finite = 0usize;
    let mut correct = 0usize;
    for ((_, target), output) in windows.iter().zip(outputs.outer_iter()) {
        if output.slice_axis(Axis(0), Slice::from(start..start + BYTE_SUPPORT_DIM)).iter().any(|value| !value.is_finite()) {
            non_finite += 1;
            continue;
        }
        let best = argmax257(output.view(), start);
        *predicted.entry(best).or_default() += 1;
        correct += usize::from(best == *target);
    }
    let mut target_counts = BTreeMap::<usize, usize>::new();
    for (_, target) in windows {
        *target_counts.entry(*target).or_default() += 1;
    }
    Ok(json!({
        "windows": windows.len(),
        // Fail closed: any non-finite output invalidates the metric.
        "top1_acc": (non_finite == 0).then(|| correct as f64 / windows.len() as f64),
        "non_finite_rows": non_finite,
        "distinct_predictions": predicted.len(),
        "unigram_acc": target_counts.values().max().copied().unwrap_or(0) as f64 / windows.len() as f64,
    }))
}

/// Promotion `TypedMetrics` grouping and pass rule (grouped Brier <= 0.2, rank accuracy >= 0.6).
#[derive(Default)]
struct TypedMetrics {
    records: usize,
    rank_correct: usize,
    brier: f64,
    /// Records with a non-finite prediction or an unnormalizable distribution.
    non_finite: usize,
}

impl TypedMetrics {
    fn observe(&mut self, kind: &str, rows: &[(f32, f32, usize)]) {
        self.records += 1;
        if rows.iter().any(|row| !row.0.is_finite() || !row.1.is_finite()) {
            self.non_finite += 1;
            return;
        }
        if kind == "noul" {
            let (predicted, target, _) = rows[0];
            let p = f64::from(predicted).clamp(1.0e-7, 1.0 - 1.0e-7);
            let t = f64::from(target);
            self.brier += (p - t).powi(2);
            self.rank_correct += usize::from((p >= 0.5) == (t >= 0.5));
            return;
        }
        let predicted_sum: f64 = rows.iter().map(|row| f64::from(row.0)).sum();
        if !predicted_sum.is_finite() || predicted_sum <= 0.0 {
            self.non_finite += 1;
            return;
        }
        let target_sum: f64 = rows.iter().map(|row| f64::from(row.1)).sum();
        let predicted_top = rows.iter().map(|row| row.0).fold(f32::NEG_INFINITY, f32::max);
        let target_top = rows.iter().map(|row| row.1).fold(f32::NEG_INFINITY, f32::max);
        let target_ties = rows.iter().filter(|row| row.1 == target_top).count();
        let predicted_ties = rows.iter().filter(|row| row.0 == predicted_top).count();
        let top_matches = rows.iter().any(|row| row.0 == predicted_top && row.1 == target_top);
        self.rank_correct += usize::from(top_matches && (target_ties > 1 || predicted_ties == 1));
        let brier: f64 = rows
            .iter()
            .map(|&(predicted, target, _)| {
                (f64::from(predicted) / predicted_sum - f64::from(target) / target_sum).powi(2)
            })
            .sum();
        self.brier += brier / rows.len() as f64;
    }
}

fn heldout_retention(
    model: Model<'_>,
    config: &MaskedPcnConfig,
    examples: &[TaskTrainingExample],
) -> Result<Value, Error> {
    let rows: Vec<Vec<f32>> = examples.iter().map(|example| example.input.to_vec()).collect();
    let outputs = predict_rows(model, config, &rows)?;
    let mut groups = BTreeMap::<(String, u64, String), Vec<(f32, f32, usize)>>::new();
    let mut sequence = (0usize, 0usize, 0usize);
    for (example, output) in examples.iter().zip(outputs.outer_iter()) {
        match &example.supervision {
            TaskSupervision::Typed { probability, .. } => {
                groups
                    .entry((example.dataset_id.clone(), example.record_id, example.task_kind.clone()))
                    .or_default()
                    .push((
                        decode_noul_state(output[GENERIC_NOUL_INDEX]),
                        *probability,
                        example.candidate_ordinal.unwrap_or(0),
                    ));
            }
            TaskSupervision::Token { token, .. } => {
                sequence.0 += 1;
                let support = output.slice_axis(Axis(0), Slice::from(TOKEN_SUPPORT_START..TOKEN_SUPPORT_START + BYTE_SUPPORT_DIM));
                if support.iter().any(|value| !value.is_finite()) {
                    sequence.2 += 1;
                } else {
                    sequence.1 += usize::from(argmax257(output, TOKEN_SUPPORT_START) == *token);
                }
            }
        }
    }
    let mut by_kind = BTreeMap::<String, TypedMetrics>::new();
    let mut total = TypedMetrics::default();
    for ((_, _, kind), rows) in &groups {
        by_kind.entry(kind.clone()).or_default().observe(kind, rows);
        total.observe(kind, rows);
    }
    // Fail closed: metrics are null when any record was non-finite.
    let report = |metrics: &TypedMetrics| {
        let records = metrics.records.max(1) as f64;
        let valid = metrics.non_finite == 0 && metrics.records > 0;
        json!({
            "records": metrics.records,
            "non_finite_records": metrics.non_finite,
            "rank_accuracy": valid.then(|| metrics.rank_correct as f64 / records),
            "brier": valid.then(|| metrics.brier / records).filter(|value| value.is_finite()),
        })
    };
    let mut value = json!({
        "rows": examples.len(),
        "typed": report(&total),
        "typed_by_kind": by_kind.iter().map(|(kind, metrics)| (kind.clone(), report(metrics))).collect::<BTreeMap<_, _>>(),
    });
    if total.records > 0 && total.non_finite == 0 {
        let records = total.records as f64;
        value["typed_promotion_rule_pass"] =
            json!(total.brier / records <= 0.2 && total.rank_correct as f64 / records >= 0.6);
    }
    value["sequence_rows"] = json!(sequence.0);
    value["sequence_non_finite_rows"] = json!(sequence.2);
    value["sequence_token_accuracy"] =
        json!((sequence.0 > 0 && sequence.2 == 0).then(|| sequence.1 as f64 / sequence.0 as f64));
    Ok(value)
}

fn parse_rates(value: &str) -> Result<Vec<f32>, Error> {
    let rates: Vec<f32> = value.split(',').map(|part| part.trim().parse()).collect::<Result<_, _>>()?;
    if rates.len() != 3 || rates.iter().any(|rate| !rate.is_finite() || *rate <= 0.0) {
        return Err("layer alphas must be three positive rates".into());
    }
    Ok(rates)
}

fn role_config(
    metadata: &pcn::UniversalCheckpointMetadata,
    role: usize,
    eta: f32,
) -> MaskedPcnConfig {
    MaskedPcnConfig {
        relax_steps: metadata.masked_pcn.relax_steps,
        alpha: metadata.masked_pcn.alpha,
        layer_alphas: metadata
            .expert_layer_alphas
            .as_ref()
            .map_or_else(|| metadata.masked_pcn.layer_alphas.clone(), |profiles| profiles[role].to_vec()),
        eta,
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Error> {
    let args = Args::parse();
    let started = Instant::now();
    fs::create_dir_all(&args.out_dir)?;
    let mut logger = Logger {
        file: fs::File::create(args.out_dir.join(format!("{}.jsonl", args.label)))?,
        label: args.label.clone(),
    };
    let token_path = args.path == OutputPath::RequestToken;
    let items = load_fit_set(&args.fit)?;
    let grouped = fit_training_examples(&items, &args.out_dir.join(format!("{}-records", args.label)), args.seed)?;
    let train_ids: Option<BTreeSet<&str>> = args.train_ids.as_deref().map(|ids| ids.split(',').map(str::trim).collect());
    if let Some(ids) = &train_ids {
        if let Some(missing) = ids.iter().find(|id| !items.iter().any(|item| item.id == **id)) {
            return Err(format!("--train-ids names unknown item {missing}").into());
        }
    }
    let selected = |item: &FitItem| train_ids.as_ref().is_none_or(|ids| ids.contains(item.id.as_str()));
    let train_items: Vec<FitItem> = items.iter().filter(|item| selected(item)).cloned().collect();
    let task_rows: Vec<TaskTrainingExample> = items
        .iter()
        .zip(grouped)
        .filter(|(item, _)| selected(item))
        .flat_map(|(_, rows)| rows)
        .collect();

    let loaded = load_universal_checkpoint(&args.checkpoint)?;
    let request_scope = args.update_scope.map_or(token_path, |scope| scope == UpdateScope::Request);
    let role = usize::from(request_scope);
    let mut cpu = loaded.pcn;
    let metadata = loaded.metadata;
    let production_eta = if request_scope { metadata.masked_pcn.eta } else { 1.0e-7 };
    let production = role_config(&metadata, role, production_eta);
    let mut config = production.clone();
    if let Some(steps) = args.relax_steps {
        config.relax_steps = steps;
    }
    if let Some(alpha) = args.alpha {
        config.alpha = alpha;
    }
    if let Some(rates) = &args.layer_alphas {
        config.layer_alphas = parse_rates(rates)?;
    }
    if let Some(eta) = args.eta {
        config.eta = eta;
    }
    let encoding = args.byte_target_encoding.unwrap_or(metadata.byte_target_encoding);
    let seal_config: SealConfig = metadata.seal.clone().ok_or("checkpoint has no SEAL config")?;
    let mut surprise: SurpriseState = metadata.surprise_state.clone().ok_or("checkpoint has no SEAL state")?;
    let guard = MaskedEnergyGuard { max_energy: args.max_energy, max_relax_steps: args.max_relax_steps };

    // Production batch construction, fixed order (the set is fixed).
    let batches: Vec<MaskedBatch> = if token_path {
        task_batches(&task_rows, args.task_batch_size, encoding)
            .map(|batch| batch.map(|(_, batch)| batch))
            .collect::<Result<_, _>>()?
    } else {
        let responses = task_rows
            .iter()
            .map(|row| prepared_response_byte_example(row, encoding)?.ok_or_else(|| "fit row not a prepared response".into()))
            .collect::<Result<Vec<_>, Error>>()?;
        responses
            .chunks(args.corpus_batch_size)
            .map(|chunk| -> Result<MaskedBatch, Error> {
                let mut inherited = make_masked_batch(chunk)?;
                let scale = chunk.len() as f32 / args.byte_head_reference_batch_size as f32 * args.byte_head_extra;
                for value in inherited.output_update_scale.iter_mut().skip(BYTE_OUTPUT_OFFSET) {
                    *value *= scale;
                }
                Ok(lift_multimodal_batch(&inherited)?)
            })
            .collect::<Result<_, _>>()?
    };

    let device = NdArrayDevice::Cpu;
    let silenced = args
        .silence_top_columns
        .as_deref()
        .map(|range| -> Result<std::ops::Range<usize>, Error> {
            let (start, end) = range.split_once("..").ok_or("silence range must be a..b")?;
            let range = start.trim().parse::<usize>()?..end.trim().parse::<usize>()?;
            if range.is_empty() || range.end > cpu.w[3].ncols() {
                return Err("silence range out of bounds".into());
            }
            Ok(range)
        })
        .transpose()?;
    let mut gpu = GpuPcn::<Cpu>::from_cpu(&cpu, &device);
    let windows = retention_windows(&args.retention_text, args.retention_windows)?;
    let typed = load_tagged_task_dataset(&args.typed_dataset, &args.typed_dataset_id, "typed-decision", 0, 1, 0)?.heldout;
    let sequence = load_tagged_task_dataset(&args.sequence_dataset, &args.sequence_dataset_id, "instruction-response", 0, 1, 0)?.heldout;
    logger.log("header", json!({
        "checkpoint": args.checkpoint, "epoch": metadata.epoch, "cumulative_batches": metadata.cumulative_batches,
        "path": format!("{:?}", args.path), "decoder": format!("{:?}", args.decoder),
        "update_scope": if request_scope { "request" } else { "inherited" },
        "silenced_top_columns": silenced.as_ref().map(|range| [range.start, range.end]),
        "fit_items": items.len(), "fit_rows": task_rows.len(), "batches": batches.len(),
        "batch_rows": batches.iter().map(|batch| batch.clean_input.nrows()).collect::<Vec<_>>(),
        "production_config": {"relax_steps": production.relax_steps, "alpha": production.alpha,
            "layer_alphas": production.layer_alphas, "eta": production.eta},
        "config": {"relax_steps": config.relax_steps, "alpha": config.alpha,
            "layer_alphas": config.layer_alphas, "eta": config.eta},
        "base_eta": args.base_eta, "byte_target_encoding": encoding.as_str(),
        "checkpoint_byte_target_encoding": metadata.byte_target_encoding.as_str(),
        "seal": args.seal, "seal_modulation": surprise.last_modulation,
        "byte_head_scale": (!token_path).then(|| batches[0].output_update_scale[BYTE_OUTPUT_OFFSET]),
        "max_energy": args.max_energy, "max_relax_steps": args.max_relax_steps,
        "typed_heldout_rows": typed.len(), "sequence_heldout_rows": sequence.len(),
        "retention_windows": windows.len(),
        "train_ids": train_items.iter().map(|item| item.id.as_str()).collect::<Vec<_>>(),
        "train_subset": args.train_ids.is_some(),
    }))?;

    // Retention before. `production` is today's served settle profile. The gate
    // compares against it; when the run overrides the settle profile, the
    // run-config baseline is also logged so the learning effect is separable.
    let backend = args.retention_backend;
    let mut heldout_rows = typed.clone();
    heldout_rows.extend(sequence.iter().cloned());
    // Inherited-scope runs never update the request expert that serves typed and
    // token outputs: measure it once with its production profile.
    let untouched_heldout = if request_scope {
        None
    } else {
        let typed_root = args.typed_checkpoint.clone().unwrap_or_else(|| {
            args.checkpoint.parent().unwrap_or_else(|| Path::new(".")).join("request-conditioned")
        });
        let typed_loaded = load_universal_checkpoint(&typed_root)?;
        let typed_config = role_config(&typed_loaded.metadata, 1, typed_loaded.metadata.masked_pcn.eta);
        let typed_gpu = (backend == Decoder::Burn).then(|| GpuPcn::<Cpu>::from_cpu(&typed_loaded.pcn, &device));
        let typed_model = typed_gpu.as_ref().map_or(Model::Native(&typed_loaded.pcn), Model::Burn);
        let mut value = heldout_retention(typed_model, &typed_config, &heldout_rows)?;
        value["serving_expert"] = json!(typed_root);
        value["unchanged_by_construction"] = json!(true);
        Some(value)
    };
    let measure = |cpu: &PCN, gpu: &GpuPcn<Cpu>, settle: &MaskedPcnConfig| -> Result<Value, Error> {
        let phase = Instant::now();
        let prose = prose_retention(Model::of(backend, cpu, gpu), settle, &windows, token_path)?;
        let heldout = match &untouched_heldout {
            Some(value) => value.clone(),
            None => heldout_retention(Model::of(backend, cpu, gpu), settle, &heldout_rows)?,
        };
        Ok(json!({"prose": prose, "heldout": heldout, "seconds": phase.elapsed().as_secs_f64()}))
    };
    let settle_changed = config.relax_steps != production.relax_steps
        || config.alpha.to_bits() != production.alpha.to_bits()
        || config.layer_alphas != production.layer_alphas
        || silenced.is_some();
    // Today's served behavior: production profile, unmodified weights.
    let production_baseline = if settle_changed {
        let value = measure(&cpu, &gpu, &production)?;
        logger.log("retention", json!({"phase": "before", "settle": "production", "retention": value}))?;
        Some(value)
    } else {
        None
    };
    if let Some(range) = &silenced {
        cpu.w[3].slice_axis_mut(Axis(1), Slice::from(range.clone())).fill(0.0);
        gpu = GpuPcn::<Cpu>::from_cpu(&cpu, &device);
    }
    let before = measure(&cpu, &gpu, &config)?;
    logger.log("retention", json!({"phase": "before", "settle": "run", "retention": before}))?;
    let baseline = production_baseline.unwrap_or_else(|| before.clone());

    let mut skipped = 0usize;
    let mut consecutive_skips = 0usize;
    let mut stop_reason: Option<&str> = None;
    let mut epoch0: Option<(usize, usize)> = None;
    let mut epoch_seconds = 0.0f64;
    let mut eval_seconds = 0.0f64;
    let mut final_report = Value::Null;
    let mut final_eval_complete = false;
    let mut epoch = 0usize;
    let mut full_eval_epoch = None;
    // Separate budgets: the epoch-0 baseline evaluation, the training loop
    // (its intermediate evaluations included), and the final fixture-cap evaluation.
    let minutes = |value: f64| std::time::Duration::from_secs_f64(value * 60.0);
    let mut train_started: Option<Instant> = None;
    loop {
        let last = epoch == args.epochs || stop_reason.is_some();
        if last && full_eval_epoch == Some(epoch) {
            break;
        }
        if stop_reason == Some("baseline_eval_budget") {
            break;
        }
        if epoch % args.eval_every.max(1) == 0 || last {
            let phase = Instant::now();
            gpu.to_cpu(&mut cpu);
            let cap = if last { None } else { args.eval_max_bytes };
            let deadline = if last {
                phase + minutes(args.max_final_minutes)
            } else if let Some(train_started) = train_started {
                train_started + minutes(args.max_minutes)
            } else {
                phase + minutes(args.max_baseline_minutes)
            };
            // Before the final evaluation only the trained items are decoded.
            let eval_items: &[FitItem] = if last { &items } else { &train_items };
            let report = evaluate_generation(&cpu, &gpu, &config, args.decoder, eval_items, token_path, cap, Some(deadline))?;
            let Some(report) = report else {
                let reason = if last {
                    "final_eval_budget"
                } else if train_started.is_some() {
                    "wall_cap"
                } else {
                    "baseline_eval_budget"
                };
                logger.log("stop", json!({"epoch": epoch, "stop_reason": reason, "eval_s": phase.elapsed().as_secs_f64()}))?;
                stop_reason = Some(reason);
                if last {
                    break;
                }
                continue;
            };
            let exact = report["overall"]["exact"].as_u64().unwrap_or(0) as usize;
            let distinct = report["overall"]["distinct_outputs"].as_u64().unwrap_or(0) as usize;
            if epoch == 0 && !last {
                epoch0 = Some((exact, distinct));
            }
            if last {
                full_eval_epoch = Some(epoch);
                final_eval_complete = true;
            } else {
                eval_seconds = phase.elapsed().as_secs_f64();
            }
            let mut record = report.clone();
            record["eval_set"] = json!(if last { "all" } else { "train" });
            record["epoch"] = json!(epoch);
            record["elapsed_s"] = json!(started.elapsed().as_secs_f64());
            record["eval_s"] = json!(phase.elapsed().as_secs_f64());
            record["skipped_batches"] = json!(skipped);
            record["seal_modulation"] = json!(surprise.last_modulation);
            logger.log("eval", record)?;
            final_report = report;
            // Stop rule: from `stop_patience_epochs` on, the trained items must beat
            // the epoch-0 baseline in exact answers or distinct outputs.
            if let Some((exact0, distinct0)) = epoch0 {
                if stop_reason.is_none() && !last && epoch >= args.stop_patience_epochs
                    && exact <= exact0 && distinct <= distinct0
                {
                    stop_reason = Some("no_gain_over_epoch0");
                    continue;
                }
            }
        }
        if last {
            break;
        }
        let train_clock = *train_started.get_or_insert_with(Instant::now);
        // Hard cap: another epoch and its evaluation must fit in the training budget.
        if train_clock.elapsed().as_secs_f64() + epoch_seconds + eval_seconds > args.max_minutes * 60.0 {
            stop_reason = Some("wall_cap");
            continue;
        }
        let phase = Instant::now();
        let (mut positive, mut free) = (0.0f32, 0.0f32);
        for batch in &batches {
            if train_clock.elapsed() > minutes(args.max_minutes) {
                stop_reason = Some("wall_cap");
                break;
            }
            let seal = args.seal.then_some((&mut surprise, &seal_config));
            let metrics = if request_scope {
                train_masked_batch_gpu_request_paths_with_seal(
                    &mut gpu, batch, &config, MULTIMODAL_INPUT_DIM, args.base_eta, seal, Some(guard),
                )?
            } else {
                train_masked_batch_gpu_inherited_paths_with_seal(
                    &mut gpu, batch, &config, CONDITION_INPUT_ROWS, seal, Some(guard),
                    pcn::gpu::IdleOutputs::Free, None,
                )?
            };
            eprintln!(
                "epoch {} batch rows {}: positive {:.4e} free {:.4e}, {:.0}s",
                epoch + 1, batch.clean_input.nrows(), metrics.positive_energy, metrics.free_energy,
                phase.elapsed().as_secs_f64(),
            );
            if !metrics.positive_energy.is_finite()
                || !metrics.free_energy.is_finite()
                || metrics.positive_energy > args.max_energy
                || metrics.free_energy > args.max_energy
            {
                skipped += 1;
                consecutive_skips += 1;
            } else {
                consecutive_skips = 0;
                positive += metrics.positive_energy / batches.len() as f32;
                free += metrics.free_energy / batches.len() as f32;
            }
            if consecutive_skips >= args.stop_consecutive_skips {
                stop_reason = Some("consecutive_energy_skips");
                break;
            }
        }
        epoch += 1;
        epoch_seconds = phase.elapsed().as_secs_f64();
        logger.log("train_epoch", json!({"epoch": epoch, "train_s": epoch_seconds,
            "positive_energy": positive, "free_energy": free, "skipped_batches": skipped,
            "seal_modulation": surprise.last_modulation}))?;
    }

    // Retention after, with the run's settle profile.
    let trained = epoch > 0;
    let after = if trained {
        gpu.to_cpu(&mut cpu);
        let value = measure(&cpu, &gpu, &config)?;
        logger.log("retention", json!({"phase": "after", "settle": "run", "retention": value}))?;
        Some(value)
    } else {
        None
    };
    // Fail closed: every metric must be present and finite before and after.
    let mut checks = Vec::new();
    let mut failures = Vec::new();
    let finite = |value: &Value| value.as_f64().filter(|value| value.is_finite());
    let mut metrics: Vec<(String, Value, Value, f64, bool)> = Vec::new();
    if let Some(after) = &after {
        let pick = |root: &Value, path: &[&str]| path.iter().fold(root.clone(), |value, key| value[*key].clone());
        let mut add = |name: String, path: Vec<&str>, tolerance: f64, higher_is_better: bool| {
            metrics.push((name, pick(&baseline, &path), pick(after, &path), tolerance, higher_is_better));
        };
        add("dracula_top1".to_owned(), vec!["prose", "top1_acc"], args.max_retention_drop, true);
        add("sequence_token_accuracy".to_owned(), vec!["heldout", "sequence_token_accuracy"], args.max_retention_drop, true);
        add("typed_rank_accuracy".to_owned(), vec!["heldout", "typed", "rank_accuracy"], args.max_retention_drop, true);
        add("typed_brier".to_owned(), vec!["heldout", "typed", "brier"], args.max_brier_rise, false);
        for family in ["choice", "score", "noul"] {
            add(format!("{family}_rank_accuracy"), vec!["heldout", "typed_by_kind", family, "rank_accuracy"], args.max_retention_drop, true);
            add(format!("{family}_brier"), vec!["heldout", "typed_by_kind", family, "brier"], args.max_brier_rise, false);
        }
    }
    for (name, before_value, after_value, tolerance, higher_is_better) in metrics {
        let (Some(before_value), Some(after_value)) = (finite(&before_value), finite(&after_value)) else {
            failures.push(format!("{name}: missing or non-finite"));
            checks.push(json!({"metric": name, "before": before_value, "after": after_value, "pass": false}));
            continue;
        };
        let worsening = if higher_is_better { before_value - after_value } else { after_value - before_value };
        let pass = worsening <= tolerance;
        if !pass {
            failures.push(format!("{name}: worsened by {worsening:.4} > {tolerance}"));
        }
        checks.push(json!({"metric": name, "before": before_value, "after": after_value,
            "worsening": worsening, "tolerance": tolerance, "pass": pass}));
    }
    let retention = if !trained {
        "baseline"
    } else if failures.is_empty() {
        "holds"
    } else {
        "fails"
    };
    let overall = &final_report["overall"];
    logger.log("summary", json!({
        "epochs_trained": epoch, "stop_reason": stop_reason, "skipped_batches": skipped,
        "elapsed_s": started.elapsed().as_secs_f64(),
        "final_eval_complete": final_eval_complete,
        "exact": overall["exact"], "distinct_outputs": overall["distinct_outputs"],
        "by_kind": final_report["by_kind"], "conversation_pairs": final_report["conversation_pairs"],
        "retention": retention, "retention_baseline_settle": if settle_changed { "production" } else { "run" },
        "retention_failures": failures, "retention_checks": checks,
        "note": "Exact recall of trained items is memorization, not generalization.",
    }))?;
    Ok(())
}
