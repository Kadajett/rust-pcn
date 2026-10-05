#!/usr/bin/env bash
# B200 throughput ladder (PLAN.md Phase 1 step 7): for every hidden width W, widen the transferred checkpoint ONCE
# into a scratch root, then for every batch size B run the real trainer on the real data for a fixed number of
# batches and report samples/s, s/batch and peak GPU memory. Never writes to the source root or the live run.
#
# Usage:
#   scripts/b200_throughput_ladder.sh --source-root <checkpoint root with experts.json> \
#       --widths 16384,24576,32768 --batches 20 [--batch-sizes 512,1024] [--warmup-batches 3] \
#       [--scratch /fast-storage/river-ladder] [--bin-dir /home/kadajett/Dev/rust-pcn/target/live] \
#       [--registry datasets/training-registry-prose.json] [--replays /bulk-storage/connectome-merc/marty-continuous-20260916] \
#       [--new-unit-scale 0.3] [--seed 20261005] [--trial-timeout 900] [--target-samples-per-s 250] \
#       [--inherited-eta 0.003] [--byte-head-reference-batch-size auto] [--replay-cache auto|keep|drop] [--keep-trials]
#
# Flags:
#   --source-root   Transferred checkpoint root (experts.json + active generation + replay-cache). Read only; passed
#                   to the widen tool as --root. A (path,size,mtime) fingerprint of the whole tree is taken before and
#                   after every widen and the ladder aborts if it changed.
#   --widths        Comma list of hidden widths W (outer loop). A W equal to the source root's own width (9216 today)
#                   is a control trial: the root is hard-linked into <scratch>/widen-W without widening.
#                   --batch-sizes  Comma list of B (inner loop).
#   --batches       Measured batches per trial (after warmup).             --warmup-batches  Batches excluded (default 3).
#   --scratch       Scratch dir: widen-W/ (kept), trial-W-B/ (deleted unless --keep-trials), telemetry-W-B/ (fresh per
#                   trial), replay-cache-valid/ (see --replay-cache), ladder-W-B.log, gpu-W-B.csv, widen-W.json,
#                   ladder-results.tsv (appended). Trial roots ALWAYS live here, never under the source root and never
#                   under /bulk-storage/connectome-merc/river-universal-checkpoints/river-v8-fresh-seed20261004: that
#                   remote path equals the LOCAL live root path (audit-fable B8), so anything trained there collides
#                   with the local root on pull-back. The real widened run root must be a NEW name such as
#                   river-v8-b200-w16384.
#   --bin-dir       Holds river-pcn-widen-universal and river-pcn-train-universal.
#   --registry / --replays   Same data as the live run (same absolute paths on the remote).
#   --seed / --new-unit-scale  Forwarded to the widen tool (new units' incoming-weight init).
#   --trial-timeout Hard wall-clock cap per trial INCLUDING startup (replay load or rebuild, corpus load, GPU upload,
#                   NVRTC JIT); default 900 s.
#   --target-samples-per-s  Threshold for the suggested W (PLAN step 7: >= ~250 samples/s); default 250.
#   --inherited-eta Forwarded (default 0.003 = live run). The trainer uses the carried health.json
#                   `inherited_eta_override` instead when it is smaller (train_universal.rs:3214-3225); the ladder
#                   reads <trial root>/health.json the same way: eta_effective = min(--inherited-eta, override).
#   --byte-head-reference-batch-size  The byte-head update coefficient is eta x 32 x B / ref (tensors.rs:591-599,
#                   multimodal.rs:14, train_universal.rs:3606-3613). The live regime is 3.33e-4 x 32 x 128 / 192 =
#                   0.0071 and both overnight rollbacks happened at 0.021 and 0.064 (audit-fable B2), so `auto`
#                   (default) picks the ref that reproduces the live coefficient at this B and eta:
#                   ref = round(192 x (B/128) x (eta_effective/3.33e-4)), eta_effective as above. A number overrides.
#                   The ref and the resulting coefficient are logged per trial and the ref is the byte_head_ref column.
#   --replay-cache  What to do with <trial root>/replay-cache (the widen tool copies the source root's cache into
#                   every widened root). That shipped cache is REJECTED by the trainer on the remote: cache.rs:65-75
#                   requires the manifest's source_root to equal the canonical path of --replays (a symlinked
#                   /bulk-storage changes it) and cache.rs:323-333 keys every shard by nanosecond mtime, which
#                   tar/rsync transfer does not preserve exactly; a mismatch is ReplayCacheError::Incompatible /
#                   StaleShard and a hard trainer exit (load_replays_cached with rebuild=false,
#                   train_universal.rs:3258-3260), not a rebuild (audit-fable B5, audit-astra section 5).
#                     drop  delete it before launch; the trainer rebuilds it from the --replays shards every trial.
#                     keep  leave it (only for a cache built on this machine against this exact --replays path).
#                     auto  (default) if <scratch>/replay-cache-valid/manifest.json exists, `cp -a` that cache in;
#                           else drop. After a trial whose trainer got past replay loading (first telemetry record
#                           published: state.json or a non-empty events.jsonl; events.jsonl itself is created empty
#                           at train_universal.rs:3244 BEFORE the replay load at 3258, the first publish is 3570 after
#                           it; or the trial ended ok) the rebuilt <trial root>/replay-cache is copied ONCE to
#                           <scratch>/replay-cache-valid/ (only if its manifest.json exists) so later trials skip it.
#                   The decision is logged per trial; startup_s (launch -> first telemetry record) in the TSV and
#                   RESULT line shows the rebuild cost. The first auto trial's --trial-timeout must cover it.
#   --keep-trials   Keep trial roots and telemetry dirs. Widened roots and the source root are never deleted.
#   JEV_PCN_CUDA_LIB_DIR / CUDA env are passed through untouched (train_universal.rs ensure_compatible_cuda_runtime
#   re-execs with that lib dir prefixed to LD_LIBRARY_PATH when libnvrtc.so exists there).
#
# Idempotent: widen-W is skipped when <scratch>/widen-W/experts.json exists; a (W,B) trial is skipped when
# ladder-results.tsv already holds an `ok` row for it. Failed/timed-out trials are re-run.
#
# Trial root: hard-linked copy of the widened root (generation-* dirs via `cp -al`; experts.json and health.json via
# `cp -a`; replay-cache per --replay-cache, always a real copy because the trainer CAN append to it in place,
# cache.rs load_replays_cached:200-202 append_records). Safe because the trainer never rewrites weight files in
# place: saves write a NEW generation dir (universal_experts.rs write_generation:348, temp dir + rename), commit by
# atomically replacing experts.json (activate_generation:395) and health.json (save_run_health:579), and prune with
# remove_dir_all (prune_generations:445) which only unlinks.
#
# Exact trainer flags (live-run flags from the systemd unit, with the trial substitutions):
#   river-pcn-train-universal --dual-expert --replays R --registry G --relax-steps 100 --max-relax-steps 200
#     --inherited-layer-alphas 0.1,0.1,0.1 --request-layer-alphas 0.00005,0.07,0.00001 --byte-target-encoding zero
#     --inherited-eta ETA --task-examples-per-dataset 512 --task-rehearsal-examples-per-dataset 64
#     --focus-lane-maintenance code,structured,choice,score,noul --noul-examples-per-stage 8 --replay-examples-per-stage 8
#     --skip-request-expert-training --mask-rate 0 --positive-phase-start fresh --inherited-idle-outputs free
#     --byte-prediction-precision 1.0
#     --output <scratch>/trial-W-B --telemetry-dir <scratch>/telemetry-W-B --run-name "ladder W=.. B=.."
#     --corpus-batch-size B --byte-head-reference-batch-size <auto ref | override>
#     --examples-per-dataset max(32768, (warmup+batches+1)*B)   # enough corpus windows that every measured batch is
#                                                               # a full-size corpus batch (train_universal.rs:3591-3604:
#                                                               # stage.examples is the first group trained)
#     --checkpoint-every-batches 1000000 --checkpoint-every-stages 1000000   # no checkpoint inside the trial
#                                                               # (in-stage flags only defer to the stage boundary anyway,
#                                                               # train_universal.rs:3753,4360)
#     --generator-eval-every-batches 0                          # disables the held-out generator eval and its load
#                                                               # (train_universal.rs:176-177, 3332-3347, 3782-3783)
#     --focus-lane-records-per-stage 0                          # skips focus-lane loading (train_universal.rs:3409)
#     --focus-eval-every-stages 1000000 --sample-every-batches 1000000 --keep-generations 1
#   Per-batch sample count: the corpus loop steps stage.examples by --corpus-batch-size (train_universal.rs:3602);
#   --batch-size only governs Noul batches (3540) and --task-batch-size the request-expert task batches (3539), both
#   after the corpus phase and outside the measured window, so only --corpus-batch-size is set to B.
#
# Stop mechanism: the trainer has no signal handler and no batch cap (--epochs counts whole stages;
# train_universal.rs:3390). The ladder polls <telemetry>/events.jsonl every second (one JSON record per published
# state, appended in publish order: write_queued_states:3000-3036) and counts records whose cumulative "batch"
# advanced past the previous record (each completed corpus batch publishes status "training" with the new
# cumulative batch: 3733-3779). When count >= warmup+batches it sends SIGTERM, waits up to 10 s, then SIGKILL.
# The trial root is scratch, so a half-written temp generation does not matter.
#
# Telemetry fields read (train_universal.rs trainer_state:711-782): "batch" (cumulative batches, metadata
# counter), "samples" (cumulative examples, 729), "unix_millis" (publish time, 781), "status" (718).
#   batches  = --batches; start = record #warmup (its completion time), end = record #(warmup+batches)
#   samples  = end.samples - start.samples;  wall = (end.unix_millis - start.unix_millis)/1000
#   samples/s = samples/wall;  s/batch = wall/batches;  peak_gpu_mib = max of `nvidia-smi --query-gpu=memory.used`
#   sampled every second during the trial. state.json's own samples_per_second (736) is a session average that
#   includes warmup, so it is not used.
#
# Results: one line per trial on stdout and appended to <scratch>/ladder-results.tsv with header
#   width batch batches samples wall_s samples_per_s s_per_batch peak_gpu_mib startup_s byte_head_ref status log
# startup_s: seconds from trainer launch to its first telemetry record (replay load/rebuild, corpus load, GPU upload,
# NVRTC JIT), `-` if none appeared. An existing ladder-results.tsv with a different header aborts the ladder.
# status: ok | ok-short-batches (mean samples/batch < 0.9 B: corpus ran out, check the log) | failed | timeout.
# At the end prints the suggested width: the largest W with any batch size reaching --target-samples-per-s.
set -euo pipefail

SOURCE_ROOT=""
WIDTHS=""
BATCH_SIZES="512,1024"
BATCHES=""
WARMUP=3
SCRATCH=/fast-storage/river-ladder
BIN_DIR=/home/kadajett/Dev/rust-pcn/target/live
REGISTRY=/home/kadajett/Dev/rust-pcn/datasets/training-registry-prose.json
REPLAYS=/bulk-storage/connectome-merc/marty-continuous-20260916
NEW_UNIT_SCALE=0.3
SEED=20261005
TRIAL_TIMEOUT=900
TARGET_SPS=250
INHERITED_ETA=0.003
BYTE_HEAD_REF=auto
REPLAY_CACHE=auto
KEEP_TRIALS=0
GRACE_SECONDS=10

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --source-root) SOURCE_ROOT=$2; shift 2 ;;
        --widths) WIDTHS=$2; shift 2 ;;
        --batch-sizes) BATCH_SIZES=$2; shift 2 ;;
        --batches) BATCHES=$2; shift 2 ;;
        --warmup-batches) WARMUP=$2; shift 2 ;;
        --scratch) SCRATCH=$2; shift 2 ;;
        --bin-dir) BIN_DIR=$2; shift 2 ;;
        --registry) REGISTRY=$2; shift 2 ;;
        --replays) REPLAYS=$2; shift 2 ;;
        --new-unit-scale) NEW_UNIT_SCALE=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --trial-timeout) TRIAL_TIMEOUT=$2; shift 2 ;;
        --target-samples-per-s) TARGET_SPS=$2; shift 2 ;;
        --inherited-eta) INHERITED_ETA=$2; shift 2 ;;
        --byte-head-reference-batch-size) BYTE_HEAD_REF=$2; shift 2 ;;
        --replay-cache) REPLAY_CACHE=$2; shift 2 ;;
        --keep-trials) KEEP_TRIALS=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown flag: $1" >&2; usage 2 ;;
    esac
done

[ -n "$SOURCE_ROOT" ] && [ -n "$WIDTHS" ] && [ -n "$BATCHES" ] || { echo "--source-root, --widths and --batches are required" >&2; usage 2; }
case "$REPLAY_CACHE" in auto|keep|drop) ;; *) echo "--replay-cache must be auto, keep or drop" >&2; exit 2 ;; esac
case "$BYTE_HEAD_REF" in auto) ;; ''|*[!0-9]*|0) echo "--byte-head-reference-batch-size must be auto or a positive integer" >&2; exit 2 ;; esac
[ -f "$SOURCE_ROOT/experts.json" ] || { echo "no experts.json in $SOURCE_ROOT" >&2; exit 2; }
WIDEN_BIN=$BIN_DIR/river-pcn-widen-universal
TRAIN_BIN=$BIN_DIR/river-pcn-train-universal
[ -x "$WIDEN_BIN" ] || { echo "missing $WIDEN_BIN" >&2; exit 2; }
[ -x "$TRAIN_BIN" ] || { echo "missing $TRAIN_BIN" >&2; exit 2; }
[ -f "$REGISTRY" ] || { echo "missing registry $REGISTRY" >&2; exit 2; }
[ -d "$REPLAYS" ] || { echo "missing replays $REPLAYS" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 2; }
SOURCE_ROOT=$(realpath "$SOURCE_ROOT")
SCRATCH=$(realpath -m "$SCRATCH")
case "$SCRATCH/" in "$SOURCE_ROOT"/*) echo "--scratch must not lie inside --source-root" >&2; exit 2 ;; esac
mkdir -p "$SCRATCH"
RESULTS=$SCRATCH/ladder-results.tsv
HEADER=$'width\tbatch\tbatches\tsamples\twall_s\tsamples_per_s\ts_per_batch\tpeak_gpu_mib\tstartup_s\tbyte_head_ref\tstatus\tlog'
if [ -f "$RESULTS" ]; then
    [ "$(head -n 1 "$RESULTS")" = "$HEADER" ] || { echo "$RESULTS has a different column layout; move it away" >&2; exit 2; }
else
    printf '%s\n' "$HEADER" > "$RESULTS"
fi
VALID_CACHE=$SCRATCH/replay-cache-valid

now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(now) $*"; }
fingerprint() { find "$1" -printf '%p %s %T@\n' | LC_ALL=C sort | sha256sum | cut -d' ' -f1; }

TRAINER_PID=""
SAMPLER_PID=""
stop_trial_processes() {
    if [ -n "$TRAINER_PID" ] && kill -0 "$TRAINER_PID" 2>/dev/null; then
        kill -TERM "$TRAINER_PID" 2>/dev/null || true
        for _ in $(seq "$GRACE_SECONDS"); do kill -0 "$TRAINER_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL "$TRAINER_PID" 2>/dev/null || true
    fi
    [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$TRAINER_PID" "$SAMPLER_PID" 2>/dev/null || true
    TRAINER_PID=""; SAMPLER_PID=""
}
trap stop_trial_processes EXIT

# progress_records <events.jsonl>: number of records whose cumulative batch advanced (= completed batches).
progress_records() {
    python3 - "$1" <<'PY'
import json, sys
count, last = 0, None
try:
    lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    lines = []
for line in lines:
    try:
        batch = json.loads(line).get("batch")
    except ValueError:
        continue
    if isinstance(batch, int):
        if last is not None and batch > last:
            count += 1
        last = batch
print(count)
PY
}

# measure <events.jsonl> <warmup> <batches> <B> -> "batches samples wall_s samples_per_s s_per_batch short_flag"
measure() {
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
path, warmup, batches, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
records, last = [], None
for line in open(path, encoding="utf-8", errors="replace").read().splitlines():
    try:
        record = json.loads(line)
    except ValueError:
        continue
    batch = record.get("batch")
    if not isinstance(batch, int):
        continue
    if last is None:
        records.append(record)  # the pre-training snapshot, published after corpus loading
    elif batch > last:
        records.append(record)
    last = batch
# records[0] = pre-training state; records[k] = completion of batch k.
if len(records) <= warmup + batches:
    sys.exit("only %d completed batches in %s, need %d" % (max(len(records) - 1, 0), path, warmup + batches))
start, end = records[warmup], records[warmup + batches]
samples = int(end["samples"]) - int(start["samples"])
wall = (int(end["unix_millis"]) - int(start["unix_millis"])) / 1000.0
if wall <= 0:
    sys.exit("non-positive wall time between records %d and %d" % (warmup, warmup + batches))
short = 1 if samples / batches < 0.9 * size else 0
print("%d %d %.3f %.2f %.3f %d" % (batches, samples, wall, samples / wall, wall / batches, short))
PY
}

result_exists() { awk -F'\t' -v w="$1" -v b="$2" '$1==w && $2==b && $11 ~ /^ok/ {found=1} END {exit !found}' "$RESULTS"; }

append_row() {  # width batch batches samples wall samples/s s/batch peak startup_s byte_head_ref status log
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$RESULTS"
    printf 'RESULT width=%s batch=%s batches=%s samples=%s wall_s=%s samples/s=%s s/batch=%s peak_gpu_mib=%s startup_s=%s byte_head_ref=%s status=%s log=%s\n' "$@"
}

# startup_seconds <events.jsonl> <launch_unix_millis> -> seconds to the first telemetry record, or "-".
startup_seconds() {
    python3 - "$1" "$2" <<'PY'
import json, sys
path, launch = sys.argv[1], int(sys.argv[2])
try:
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    lines = []
for line in lines:
    try:
        millis = json.loads(line).get("unix_millis")
    except ValueError:
        continue
    if isinstance(millis, int):
        print("%.1f" % ((millis - launch) / 1000.0))
        break
else:
    print("-")
PY
}

# byte_head_reference <trial_root> <B> <--inherited-eta> <auto|N> -> "ref eta_effective coefficient"
# eta_effective = min(--inherited-eta, health.json inherited_eta_override) exactly as train_universal.rs:3214-3225;
# auto ref = round(192 * (B/128) * (eta_effective/3.33e-4)) reproduces the live coefficient 0.0071.
byte_head_reference() {
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
trial, size, eta, explicit = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
try:
    override = json.load(open(trial + "/health.json", encoding="utf-8")).get("inherited_eta_override")
except (OSError, ValueError, AttributeError):
    override = None
effective = eta
if isinstance(override, (int, float)) and not isinstance(override, bool) and 0 < override < eta:
    effective = float(override)
ref = int(explicit) if explicit != "auto" else max(1, round(192 * (size / 128) * (effective / 3.33e-4)))
print("%d %g %.4f" % (ref, effective, effective * 32 * size / ref))
PY
}

source_hidden_width() {  # hidden width of the source root's active generation (dims[1] of its inherited checkpoint)
    python3 - "$SOURCE_ROOT" <<'PY'
import json, sys
root = sys.argv[1]
manifest = json.load(open(f"{root}/experts.json"))
dims = manifest.get("dimensions") or json.load(open(f"{root}/{manifest['experts'][0]['checkpoint']}/checkpoint.json"))["dimensions"]
print(dims[1])
PY
}

widen() {  # W -> widened root path (stdout); W equal to the source width = control trial, no widening
    local width=$1 root="$SCRATCH/widen-$1"
    if [ -f "$root/experts.json" ]; then
        log "widen-$width exists, skipping widen" >&2
        echo "$root"; return
    fi
    if [ "$width" = "$(source_hidden_width)" ]; then
        log "control: hidden $width equals the source width; hard-linking $SOURCE_ROOT -> $root (no widen)" >&2
        make_trial_root "$SOURCE_ROOT" "$root"
        [ ! -d "$SOURCE_ROOT/replay-cache" ] || cp -a "$SOURCE_ROOT/replay-cache" "$root/replay-cache"
        echo "$root"; return
    fi
    local before after
    before=$(fingerprint "$SOURCE_ROOT")
    ls -la --time-style=full-iso "$SOURCE_ROOT" "$SOURCE_ROOT/experts.json" > "$SCRATCH/source-before-$width.txt"
    log "widening $SOURCE_ROOT -> $root (hidden $width, seed $SEED, new-unit-scale $NEW_UNIT_SCALE)" >&2
    rm -rf "$root" "$root.partial"
    "$WIDEN_BIN" --root "$SOURCE_ROOT" --hidden "$width" --output "$root.partial" --seed "$SEED" \
        --new-unit-scale "$NEW_UNIT_SCALE" --experts inherited > "$SCRATCH/widen-$width.json"
    mv "$root.partial" "$root"
    ls -la --time-style=full-iso "$SOURCE_ROOT" "$SOURCE_ROOT/experts.json" > "$SCRATCH/source-after-$width.txt"
    after=$(fingerprint "$SOURCE_ROOT")
    if [ "$before" != "$after" ]; then
        diff "$SCRATCH/source-before-$width.txt" "$SCRATCH/source-after-$width.txt" >&2 || true
        echo "ABORT: source root $SOURCE_ROOT changed during widen (fingerprint $before -> $after)" >&2
        exit 3
    fi
    log "widen-$width done, source root untouched (fingerprint $before); summary in $SCRATCH/widen-$width.json" >&2
    echo "$root"
}

make_trial_root() {  # widened_root trial_root   (replay-cache is handled by stage_replay_cache)
    local src=$1 dst=$2 entry name
    rm -rf "$dst"; mkdir -p "$dst"
    for entry in "$src"/* "$src"/.[!.]*; do
        [ -e "$entry" ] || continue
        name=$(basename "$entry")
        case "$name" in
            generation-*) cp -al "$entry" "$dst/$name" ;;   # weight files are never rewritten in place
            replay-cache) ;;
            *) cp -a "$entry" "$dst/$name" ;;              # experts.json, health.json
        esac
    done
}

stage_replay_cache() {  # widened_root trial_root -> logs the --replay-cache decision; trial root has no cache yet
    local src=$1 dst=$2
    case "$REPLAY_CACHE" in
        keep)
            if [ -d "$src/replay-cache" ]; then
                cp -a "$src/replay-cache" "$dst/replay-cache"
                log "replay-cache keep: copied $src/replay-cache (trainer exits if its source_root/mtimes mismatch)"
            else
                log "replay-cache keep: $src has no replay-cache; trainer rebuilds from $REPLAYS"
            fi ;;
        drop) log "replay-cache drop: shipped cache not staged; trainer rebuilds from $REPLAYS" ;;
        auto)
            if [ -f "$VALID_CACHE/manifest.json" ]; then
                cp -a "$VALID_CACHE" "$dst/replay-cache"
                log "replay-cache auto: copied $VALID_CACHE (built on this machine), no rebuild expected"
            else
                log "replay-cache auto: no $VALID_CACHE yet, shipped cache dropped; trainer rebuilds from $REPLAYS"
            fi ;;
    esac
}

save_replay_cache() {  # trial_root: keep the first cache the trainer rebuilt here for the remaining trials
    local trial=$1
    [ "$REPLAY_CACHE" = auto ] || return 0
    [ -f "$VALID_CACHE/manifest.json" ] && return 0
    [ -f "$trial/replay-cache/manifest.json" ] || { log "replay-cache auto: $trial/replay-cache has no manifest.json, not saved"; return 0; }
    rm -rf "$VALID_CACHE.partial"
    cp -a "$trial/replay-cache" "$VALID_CACHE.partial" && mv "$VALID_CACHE.partial" "$VALID_CACHE"
    log "replay-cache auto: saved rebuilt cache to $VALID_CACHE ($(du -sh "$VALID_CACHE" | cut -f1))"
}

run_trial() {  # width widened_root batch
    local width=$1 wroot=$2 size=$3
    local trial="$SCRATCH/trial-$width-$size" telemetry="$SCRATCH/telemetry-$width-$size"
    local logfile="$SCRATCH/ladder-$width-$size.log" gpulog="$SCRATCH/gpu-$width-$size.csv"
    local ref eta_effective coefficient needed examples status peak startup
    needed=$(( (WARMUP + BATCHES + 1) * size ))
    examples=$(( needed > 32768 ? needed : 32768 ))
    log "trial W=$width B=$size: preparing $trial"
    make_trial_root "$wroot" "$trial"
    stage_replay_cache "$wroot" "$trial"
    read -r ref eta_effective coefficient <<< "$(byte_head_reference "$trial" "$size" "$INHERITED_ETA" "$BYTE_HEAD_REF")"
    log "trial W=$width B=$size: byte-head ref $ref ($BYTE_HEAD_REF), eta_effective $eta_effective, coefficient eta*32*B/ref = $coefficient (live 0.0071)"
    rm -rf "$telemetry"; mkdir -p "$telemetry"
    : > "$logfile"; : > "$gpulog"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$gpulog" 2>/dev/null &
    SAMPLER_PID=$!
    local started deadline launch_ms
    started=$(date +%s); deadline=$((started + TRIAL_TIMEOUT)); launch_ms=$(date +%s%3N)
    log "trial W=$width B=$size: launching trainer (log $logfile, timeout ${TRIAL_TIMEOUT}s)"
    "$TRAIN_BIN" --dual-expert --replays "$REPLAYS" --registry "$REGISTRY" \
        --relax-steps 100 --max-relax-steps 200 \
        --inherited-layer-alphas 0.1,0.1,0.1 --request-layer-alphas 0.00005,0.07,0.00001 \
        --byte-target-encoding zero --inherited-eta "$INHERITED_ETA" \
        --task-examples-per-dataset 512 --task-rehearsal-examples-per-dataset 64 \
        --focus-lane-maintenance code,structured,choice,score,noul \
        --noul-examples-per-stage 8 --replay-examples-per-stage 8 \
        --skip-request-expert-training --mask-rate 0 --positive-phase-start fresh \
        --inherited-idle-outputs free --byte-prediction-precision 1.0 \
        --output "$trial" --telemetry-dir "$telemetry" --run-name "ladder W=$width B=$size" \
        --corpus-batch-size "$size" --byte-head-reference-batch-size "$ref" \
        --examples-per-dataset "$examples" \
        --checkpoint-every-batches 1000000 --checkpoint-every-stages 1000000 \
        --generator-eval-every-batches 0 --focus-lane-records-per-stage 0 \
        --focus-eval-every-stages 1000000 --sample-every-batches 1000000 --keep-generations 1 \
        > "$logfile" 2>&1 &
    TRAINER_PID=$!
    local target=$((WARMUP + BATCHES)) completed=0 exited=0 past_replay=0
    status=ok
    while :; do
        if ! kill -0 "$TRAINER_PID" 2>/dev/null; then exited=1; break; fi
        if [ "$past_replay" = 0 ] && { [ -s "$telemetry/events.jsonl" ] || [ -f "$telemetry/state.json" ]; }; then
            past_replay=1
            log "trial W=$width B=$size: first telemetry record after $(( $(date +%s) - started )) s (replay + corpus loaded)"
        fi
        completed=$(progress_records "$telemetry/events.jsonl")
        if [ "$completed" -ge "$target" ]; then break; fi
        if [ "$(date +%s)" -ge "$deadline" ]; then status=timeout; break; fi
        sleep 1
    done
    local exit_code=0
    if [ "$exited" = 1 ]; then
        wait "$TRAINER_PID" || exit_code=$?
        TRAINER_PID=""
        if [ "$completed" -lt "$target" ]; then
            completed=$(progress_records "$telemetry/events.jsonl")
            [ "$completed" -ge "$target" ] || status=failed
        fi
        [ "$status" = ok ] || log "trial W=$width B=$size: trainer exited early (code $exit_code) after $completed batches"
    fi
    stop_trial_processes
    if [ -s "$telemetry/events.jsonl" ] || [ -f "$telemetry/state.json" ]; then past_replay=1; fi
    peak=$(awk 'NF && $1+0==$1 {if ($1+0 > m) m=$1+0} END {print m+0}' "$gpulog")
    startup=$(startup_seconds "$telemetry/events.jsonl" "$launch_ms")
    log "trial W=$width B=$size: startup ${startup}s from launch to first telemetry record"
    local row
    if [ "$status" = ok ] && row=$(measure "$telemetry/events.jsonl" "$WARMUP" "$BATCHES" "$size" 2>>"$logfile"); then
        local batches samples wall sps spb short
        read -r batches samples wall sps spb short <<< "$row"
        [ "$short" = 1 ] && status=ok-short-batches
        append_row "$width" "$size" "$batches" "$samples" "$wall" "$sps" "$spb" "$peak" "$startup" "$ref" "$status" "$logfile"
    else
        [ "$status" = ok ] && status=failed
        append_row "$width" "$size" "$completed" "-" "-" "-" "-" "$peak" "$startup" "$ref" "$status" "$logfile"
        echo "--- last log lines ($logfile) ---"
        tail -n 15 "$logfile" || true
        echo "---"
    fi
    if [ "$past_replay" = 1 ] || [ "${status#ok}" != "$status" ]; then
        save_replay_cache "$trial"
    elif [ "$REPLAY_CACHE" = auto ] && [ ! -f "$VALID_CACHE/manifest.json" ]; then
        log "replay-cache auto: trainer never published telemetry, $trial/replay-cache not trusted and not saved"
    fi
    if [ "$KEEP_TRIALS" = 0 ]; then
        rm -rf "$trial" "$telemetry"
        log "trial W=$width B=$size: removed $trial and $telemetry"
    fi
}

log "ladder: source $SOURCE_ROOT widths $WIDTHS batch sizes $BATCH_SIZES warmup $WARMUP measured $BATCHES scratch $SCRATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
IFS=',' read -r -a WIDTH_LIST <<< "$WIDTHS"
IFS=',' read -r -a SIZE_LIST <<< "$BATCH_SIZES"
for width in "${WIDTH_LIST[@]}"; do
    widened=$(widen "$width")
    for size in "${SIZE_LIST[@]}"; do
        if result_exists "$width" "$size"; then
            log "trial W=$width B=$size already ok in $RESULTS, skipping"
            continue
        fi
        run_trial "$width" "$widened" "$size"
    done
done

echo
echo "=== ladder results ($RESULTS) ==="
column -t -s $'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
suggestion=$(awk -F'\t' -v target="$TARGET_SPS" -v widths="$WIDTHS" 'BEGIN { n = split(widths, want, ",") }
    NR > 1 && $11 ~ /^ok/ && $6 != "-" { for (i = 1; i <= n; i++) if ($1 == want[i] && $6 + 0 >= target && ($1 + 0 > bw || ($1 + 0 == bw && $6 + 0 > bs))) { bw = $1 + 0; bs = $6; bb = $2 } }
    END { if (bw) printf "W=%d (batch %d, %s samples/s)", bw, bb, bs }' "$RESULTS")
if [ -n "$suggestion" ]; then
    echo "SUGGESTED: largest width with >= $TARGET_SPS samples/s among $WIDTHS: $suggestion"
else
    echo "SUGGESTED: no width in $WIDTHS reached $TARGET_SPS samples/s; pick the smallest width or lower the target"
fi
