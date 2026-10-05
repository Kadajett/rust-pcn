const $ = id => document.getElementById(id);
const history = [];
let lastState = null;
let lastEventKey = '';
let lastSamplesBatch = '';
let samplesLoading = false;
let auditHistory = [];
const NOUL_MAE_TOLERANCE = 0.1;
const NOUL_SATURATION_EPSILON = 0.0001;
const NOUL_SOFT_TARGET_MARGIN = 0.05;
// This panel shows only prose questions and the text the model generated for them.
const SAMPLE_TYPES = ['prose'];
const SAMPLE_ROWS = 20;
const SAMPLE_FETCH_LIMIT = 100;

function isProseTextSample(sample) {
  return sample.output_type === 'prose'
    && sample.aggregate !== true
    && typeof sample.input?.prompt === 'string'
    && typeof sample.predicted === 'string';
}

function proseInputText(sample) {
  const { instructions, prompt } = sample.input;
  return instructions ? `${instructions}\n${prompt}` : prompt;
}

async function loadFridayUpdate() {
  try {
    const response = await fetch('/api/friday-update', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const update = await response.json();
    const timestamp = Date.parse(update.updated_at);
    if (typeof update.message !== 'string' || !Number.isFinite(timestamp)) {
      throw new Error('Invalid status record');
    }
    const message = $('fridayUpdateMessage');
    if (message.textContent !== update.message) message.textContent = update.message;
    const overdue = Date.now() - timestamp > 5 * 60_000;
    $('friday-update').classList.toggle('stale', overdue);
    $('fridayUpdateTime').textContent = `Updated ${pacific(timestamp)}${overdue ? ' · update overdue' : ''}`;
  } catch (error) {
    $('friday-update').classList.add('stale');
    $('fridayUpdateTime').textContent = `Status unavailable: ${error.message}`;
  }
}

function capabilityText(tag, text, className = '') {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function capabilityValue(value) {
  if (value === undefined || value === null) return 'Unavailable · not recorded';
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function capabilityMetrics(metrics, prefix = '') {
  return ['brier', 'mae'].filter(key => finite(metrics?.[`${prefix}${key}`]))
    .map(key => `${key === 'mae' ? 'MAE' : 'Brier'} ${number(metrics[`${prefix}${key}`], 6)}`);
}

function renderCapabilities(report) {
  if (report.schema !== 'river-capability-evidence-v1'
      || !Array.isArray(report.cases) || !report.cases.length
      || !report.aggregate_by_family || typeof report.aggregate_by_family !== 'object') {
    throw new Error('No complete semantic evidence report is available');
  }
  const checkpoint = report.checkpoint_state || {};
  const before = checkpoint.before_identity || {};
  const after = checkpoint.after_identity || {};
  const finished = Date.parse(report.finished_at);
  const hasGrade = item => typeof item.pass === 'boolean' && item.actual != null;
  const graded = report.cases.filter(hasGrade);
  const passed = graded.filter(item => item.pass).length;
  $('capabilityStatus').textContent = !graded.length
    ? `No graded model evidence · ${report.cases.length} outcomes unavailable`
    : graded.length === report.cases.length
      ? `${passed}/${graded.length} passed · ${percent(passed / graded.length)} on recorded fixtures`
      : `${passed}/${graded.length} graded fixtures passed · ${percent(passed / graded.length)} · ${report.cases.length - graded.length} outcomes unavailable`;
  $('capabilityScope').textContent = `Saved scope: ${report.scope || 'not recorded'} · Fixture version: ${report.fixture_version || 'not recorded'}`;
  $('capabilityFreshness').textContent = [
    Number.isFinite(finished)
      ? `Finished ${pacific(finished)} · age ${duration((Date.now() - finished) / 1000)}`
      : 'Finished time / age: not recorded',
    finite(before.batch) && finite(after.batch)
      ? `Observed live batches ${number(before.batch, 0)}–${number(after.batch, 0)}`
      : 'Observed batch range: not recorded',
    'Historical observation, not evidence of the current checkpoint',
  ].join(' · ');
  const frozen = checkpoint.frozen_verified ?? report.frozen_verified;
  const live = checkpoint.nonfrozen_live_state ?? report.nonfrozen_live_state;
  $('capabilityWarning').textContent = [
    live === true || frozen === false
      ? 'LIVE NON-FROZEN MEASUREMENT · immutable weights were not verified.'
      : frozen === true
        ? 'Frozen verification recorded by the report.'
        : 'Frozen state verification unavailable · immutable weights are not established.',
    checkpoint.warning || report.warning || 'No additional state warning was recorded.',
    !graded.length ? 'No model output was graded. Transport failures are availability errors, not model accuracy.' : '',
    'Reading this report does not run model evaluations.',
  ].join(' ');
  const beforeBody = checkpoint.before?.body || {};
  const afterBody = checkpoint.after?.body || {};
  $('capabilityContract').textContent = [
    `Recorded runtime contract (before → after): ${beforeBody.runtime_contract || 'not recorded'} → ${afterBody.runtime_contract || 'not recorded'}`,
    `Recorded Noul probability contract (before → after): ${beforeBody.noul_probability_contract || 'not recorded'} → ${afterBody.noul_probability_contract || 'not recorded'}`,
    `Recorded checkpoint (before → after): ${before.checkpoint || 'not recorded'} → ${after.checkpoint || 'not recorded'}`,
  ].join('\n');
  const summary = $('capabilitySummary');
  summary.replaceChildren();
  for (const family of ['prose', 'code', 'structured', 'choice', 'score', 'noul']) {
    const metrics = report.aggregate_by_family[family];
    const cell = document.createElement('div');
    const label = ['choice', 'score', 'noul'].includes(family)
      ? family[0].toUpperCase() + family.slice(1) : family;
    cell.append(capabilityText('span', label));
    const familyCases = report.cases.filter(item => item.family === family);
    const familyGrades = familyCases.filter(hasGrade);
    const familyPassed = familyGrades.filter(item => item.pass).length;
    const unavailable = familyCases.length - familyGrades.length;
    cell.append(capabilityText('strong', familyGrades.length
      ? `${familyPassed}/${familyGrades.length} passed · ${percent(familyPassed / familyGrades.length)}`
      : 'No graded evidence'));
    if (unavailable) cell.append(capabilityText('small', `${unavailable} outcomes unavailable`));
    if (metrics && familyGrades.length) {
      for (const metric of capabilityMetrics(metrics, 'mean_')) {
        cell.append(capabilityText('small', `Mean ${metric}`));
      }
    }
    summary.append(cell);
  }
  const cases = $('capabilityCases');
  cases.replaceChildren();
  for (const item of report.cases) {
    const card = document.createElement('article');
    card.className = 'capability-case';
    const heading = document.createElement('div');
    heading.className = 'capability-case-heading';
    heading.append(capabilityText('h3', `${item.family || 'Family not recorded'} · ${item.id || 'Fixture ID not recorded'}`));
    heading.append(capabilityText('strong',
      !hasGrade(item) ? 'Grade unavailable' : item.pass ? 'PASS' : 'FAIL',
      !hasGrade(item) ? '' : item.pass ? 'capability-pass' : 'capability-fail'));
    card.append(heading);
    const columns = document.createElement('div');
    columns.className = 'capability-columns';
    for (const [label, value] of [
      ['Expected complete answer / target', item.expected],
      ['Actual saved output', item.actual],
      ['Semantic diff (actual − expected where numeric)', item.diff],
    ]) {
      const column = document.createElement('div');
      column.append(capabilityText('h4', label), capabilityText('pre', capabilityValue(value)));
      columns.append(column);
    }
    card.append(columns);
    const grade = capabilityMetrics(item.grading).join(' · ');
    card.append(capabilityText('p', [
      grade || 'Brier / MAE: not recorded',
      Array.isArray(item.errors)
        ? item.errors.length ? `Errors: ${item.errors.join(' · ')}` : 'Errors: none recorded'
        : 'Errors: not recorded',
    ].join(' · '), 'capability-outcome'));
    const details = document.createElement('details');
    details.append(capabilityText('summary', 'Saved input, output instructions, and grading'));
    details.append(capabilityText('pre', capabilityValue({
      inputs: item.request?.inputs,
      outputs: item.request?.outputs,
      grading: item.grading,
    })));
    card.append(details);
    cases.append(card);
  }
}

async function loadCapabilities() {
  try {
    const response = await fetch('/api/capabilities', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderCapabilities(await response.json());
  } catch (error) {
    $('capabilityStatus').textContent = `Semantic evidence unavailable: ${error.message}`;
    for (const id of ['capabilityScope', 'capabilityFreshness', 'capabilityWarning', 'capabilityContract']) {
      $(id).textContent = '';
    }
    $('capabilitySummary').replaceChildren();
    $('capabilityCases').replaceChildren(capabilityText('p',
      'No readable report is available. Accuracy, family counts, and outcomes are unavailable, not zero.',
      'empty'));
  }
}

function gradeNoul(target, prediction) {
  const absoluteError = Math.abs(target - prediction);
  const squaredError = absoluteError ** 2;
  const softTarget = target > NOUL_SOFT_TARGET_MARGIN
    && target < 1 - NOUL_SOFT_TARGET_MARGIN;
  const saturated = prediction <= NOUL_SATURATION_EPSILON
    || prediction >= 1 - NOUL_SATURATION_EPSILON;
  const saturationFailure = softTarget && saturated;
  return {
    absoluteError,
    squaredError,
    saturated: saturationFailure,
    passed: absoluteError <= NOUL_MAE_TOLERANCE && !saturationFailure,
  };
}

const finite = value => value !== null && value !== undefined && value !== ''
  && Number.isFinite(Number(value));
// The owner is in San Francisco: every displayed wall-clock time is US Pacific.
// dateStyle/timeStyle cannot be combined with timeZoneName (TypeError), so spell out the parts.
const pacific = value => new Date(value).toLocaleString('en-US', {
  timeZone: 'America/Los_Angeles', month: 'short', day: 'numeric', year: 'numeric',
  hour: 'numeric', minute: '2-digit', second: '2-digit', timeZoneName: 'short',
});
const number = (value, digits = 3) => finite(value)
  ? Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
  : '—';
const compact = value => finite(value)
  ? Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(Number(value))
  : '—';
const statusName = value => String(value || 'waiting').replaceAll('_', ' ');
const scientific = value => finite(value) ? Number(value).toExponential(2) : '—';
const duration = value => {
  if (!finite(value)) return '—';
  const seconds = Math.max(0, Math.round(Number(value)));
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor(seconds % 86400 / 3600);
  const minutes = Math.floor(seconds % 3600 / 60);
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m ${seconds % 60}s`;
};
const percent = value => finite(value) ? `${number(Number(value) * 100, 1)}%` : '—';

function setState(state) {
  lastState = state;
  const status = statusName(state.status);
  if (state.runtime_contract === 'river-runtime-request-v1') {
    $('probeNote').textContent = 'Runtime v1 active · named Noul, Choice, Score, text, and structured answers run between batches.';
  } else if (['river-universal-trainer-state-v1', 'river-universal-trainer-state-v2'].includes(state.schema)) {
    $('probeNote').textContent = 'Universal output path is warming from zero-disabled coordinates; probes remain unavailable until runtime v1 is advertised.';
  } else if (typeof state.checkpoint === 'string') {
    $('probeNote').textContent = 'A prior checkpoint is active · text and structured answers remain available while the request-conditioned runtime starts.';
  } else {
    $('probeNote').textContent = 'No probe-capable runtime or checkpoint is advertised yet; no output will be inferred.';
  }
  const releaseName = state.public_release || 'River Song v0.1';
  $('status').textContent = status;
  $('phaseLabel').textContent = `${releaseName} · ${state.detail || status}`;
  $('runName').textContent = state.run || releaseName;
  const active = ['training', 'training tasks', 'checkpointing', 'loading corpora', 'loading_corpora', 'stage exhausted'].includes(status);
  $('connectionDot').className = `dot ${active ? 'live' : state.status === 'failed' || state.status === 'safety_stop' ? 'error' : ''}`;

  const batch = Number(state.batch) || 0;
  const total = Number(state.total_batches) || 0;
  const epochBatch = Number(state.epoch_batch) || 0;
  const batchesPerEpoch = Number(state.batches_per_epoch) || 0;
  const continuous = ['continuous', 'continuous_stages'].includes(state.training_mode);
  const progressCurrent = continuous ? epochBatch : batch;
  const progressTotal = continuous ? batchesPerEpoch : total;
  const progress = progressTotal ? Math.min(100, 100 * progressCurrent / progressTotal) : 0;
  $('progressBar').style.width = `${progress}%`;
  $('progressText').textContent = continuous
    ? `${batch.toLocaleString()} total batches · cycle batch ${epochBatch.toLocaleString()} / ${batchesPerEpoch.toLocaleString()} · ${number(progress, 2)}%`
    : `${batch.toLocaleString()} / ${total.toLocaleString()} batches · ${number(progress, 2)}%`;
  $('epochMetric').textContent = state.epoch ?? '—';
  const stageEpoch = Number(state.run_epoch) || 0;
  $('epochDetail').textContent = continuous
    ? `stage ${stageEpoch.toLocaleString()} · ${Number(state.scheduled_datasets?.length || 0).toLocaleString()} accepted datasets · forward/reverse traversal`
    : batchesPerEpoch
      ? `target ${state.epochs ?? '—'} · stage ${stageEpoch}/${state.run_epochs ?? '—'} · batch ${epochBatch}/${batchesPerEpoch}`
      : `target ${state.epochs ?? '—'}`;
  $('freeEnergy').textContent = number(state.mean_free_energy ?? state.free_energy, 5);
  $('positiveEnergy').textContent = number(state.mean_positive_energy ?? state.positive_energy, 5);
  $('throughput').textContent = number(state.samples_per_second, 2);
  $('sampleCount').textContent = compact(state.samples);
  $('anchorCount').textContent = `${compact(state.anchor_samples)} ${state.sample_detail || 'anchor rehearsals'}`;
  $('checkpointBatch').textContent = state.checkpoint_batch ? `batch ${Number(state.checkpoint_batch).toLocaleString()}` : 'Base';
  $('elapsedMetric').textContent = duration(state.training_elapsed_seconds);
  $('etaMetric').textContent = duration(state.eta_seconds);
  $('etaDetail').textContent = state.eta_kind === 'checkpoint'
    ? 'to next durable checkpoint'
    : 'at current batch pace';
  $('totalEtaMetric').textContent = duration(state.total_eta_seconds);
  const remainingScheduled = Number(state.remaining_scheduled_examples) || 0;
  const totalScheduled = Number(state.total_scheduled_examples) || 0;
  $('totalEtaDetail').textContent = totalScheduled
    ? `${compact(remainingScheduled)} / ${compact(totalScheduled)} examples remaining`
    : 'waiting for the active schedule';
  $('checkpointPath').textContent = state.checkpoint || 'waiting';
  $('relaxSteps').textContent = state.relax_steps ?? '—';
  $('alpha').textContent = number(state.alpha, 4);
  for (const [index, id] of ['inheritedStateRates', 'requestStateRates'].entries()) {
    const rates = state.expert_layer_alphas?.[index];
    $(id).textContent = Array.isArray(rates) && rates.length === 3
      ? rates.map(rate => scientific(rate)).join(' / ') : 'No per-expert override';
  }
  $('eta').textContent = scientific(state.eta);

  const now = Number(state.server_unix_millis) || Date.now();
  const stamp = Number(state.unix_millis);
  $('freshness').textContent = Number.isFinite(stamp)
    ? `updated ${Math.max(0, Math.round((now - stamp) / 1000))}s ago`
    : 'telemetry warming up';

  if (state.corpora) renderCorpora(state.corpora);
  if (state.dataset_registry) renderDatasetRegistry(state.dataset_registry);
  renderPromotion(state.promotion);
  renderGenerator(state);
  if (String(batch) !== lastSamplesBatch) loadSamples(batch);
  const key = `${state.epoch || 0}:${batch}:${state.status}`;
  if (key !== lastEventKey && Number.isFinite(Number(state.free_energy ?? state.mean_free_energy))) {
    lastEventKey = key;
    addHistory(state);
  }
}

function renderPromotion(promotion) {
  if (!promotion || !['river-promotion-evaluation-v1', 'river-promotion-evaluation-v2',
    'river-promotion-evaluation-v3'].includes(promotion.schema)) {
    $('promotionStatus').textContent = 'Waiting for the first task-aware evaluation.';
    $('promotionTypedDetail').textContent = '';
    for (const id of [
      'promotionTypedAccuracy',
      'promotionTypedBrier',
      'promotionSequenceAccuracy',
      'promotionVisionAccuracy',
      'promotionExamples',
      'promotionBatch',
    ]) $(id).textContent = '—';
    return;
  }
  const typed = promotion.typed || {};
  const sequence = promotion.sequence || {};
  const vision = promotion.vision_language || {};
  const total = (Number(typed.rows) || 0)
    + (Number(sequence.examples) || 0)
    + (Number(vision.examples) || 0);
  const contractChanged = lastState?.noul_probability_contract
    && promotion.noul_probability_contract !== lastState.noul_probability_contract;
  $('promotionStatus').textContent = contractChanged
    ? 'Historical typed calibration · waiting for evaluation under the active probability contract.'
    : promotion.promotion_ready
      ? 'All fixed held-out gates pass.'
      : 'Not promoted · at least one fixed held-out gate is below threshold.';
  $('promotionTypedAccuracy').textContent = `${percent(typed.rank_accuracy)} · ${contractChanged ? 'historical' : typed.pass ? 'pass' : 'hold'}`;
  const typedDetails = Object.entries(typed.by_output_type || {}).map(([type, metrics]) =>
    `${type}: rank ${percent(metrics.rank_accuracy)}, Brier ${number(metrics.brier, 4)}${finite(metrics.score_mae) ? `, score MAE ${number(metrics.score_mae, 3)}` : ''}`);
  $('promotionTypedDetail').textContent = [
    `Typed probability contract: ${promotion.noul_probability_contract || 'historical centered-target/softsign'}`,
    ...typedDetails,
  ].join(' · ');
  $('promotionTypedBrier').textContent = finite(typed.brier) ? number(typed.brier, 4) : '—';
  const frozenRequestExpert = lastState?.skip_request_expert_training === true;
  $('promotionSequenceAccuracy').textContent = `${percent(sequence.token_accuracy)} · ${sequence.pass ? 'pass' : 'hold'}`
    + (frozenRequestExpert ? ' · request expert (frozen in prose focus)' : '');
  $('promotionVisionAccuracy').textContent = `${percent(vision.token_accuracy)} · ${vision.pass ? 'pass' : 'hold'}`;
  $('promotionExamples').textContent = total.toLocaleString();
  $('promotionBatch').textContent = finite(promotion.batch)
    ? `batch ${Number(promotion.batch).toLocaleString()}`
    : '—';
}

function renderGenerator(state) {
  const heldout = state.generator_heldout;
  const free = state.generator_free_phase;
  const vs = (value, baseline) => `${percent(value)} · baseline ${percent(baseline)}`;
  const byteLabel = byte => byte === ' ' ? 'space' : typeof byte === 'string' ? `'${byte}'` : '—';
  const measured = heldout && heldout.status !== 'unavailable';
  $('generatorHeldoutTop1').textContent = measured ? vs(heldout.top1_accuracy, heldout.majority_baseline_accuracy) : '—';
  $('generatorHeldoutRank').textContent = measured
    ? `${percent(heldout.top5_accuracy)} · rank ${number(heldout.mean_rank, 1)}` : '—';
  // A space-only predictor scores 0% on non-space rows, so there is no space baseline to compare here.
  $('generatorHeldoutNonSpace').textContent = measured ? percent(heldout.non_space_accuracy) : '—';
  $('generatorHeldoutDistinct').textContent = measured
    ? `${heldout.distinct_predictions ?? '—'} · mode ${byteLabel(heldout.mode_prediction)} ${percent(heldout.mode_share)}` : '—';
  $('generatorFreeTop1').textContent = free ? vs(free.top1_accuracy, free.majority_baseline_accuracy) : '—';
  $('generatorFreeDetail').textContent = free
    ? `${percent(free.non_space_accuracy)} · ${free.distinct_predictions ?? '—'} distinct · mode ${byteLabel(free.mode_prediction)} ${percent(free.mode_share)}`
    : '—';
  const objective = state.byte_prediction;
  $('generatorObjective').textContent = objective?.enabled
    ? `context predicts next byte (λ ${number(objective.precision, 2)}) · byte error/row free ${number(objective.free_energy_per_row, 3)}`
      + ` · guided ${number(objective.positive_energy_per_row, 3)} · bias ‖c‖ ${number(objective.bias_norm, 3)}`
    : 'contrastive clamp only (no context-to-byte error term)';
  const health = state.generator_health;
  $('generatorSafety').textContent = health
    ? `${health.blocked ? 'BLOCKED' : health.status || '—'} · byte σ₁² ${number(health.bytes?.sigma1_sq, 3)} / cap ${number(health.bound?.spectral_cap, 1)}`
      + ` · rank-1 ${percent(health.bytes?.rank1_share)} · cap hits ${health.bound?.hits_in_session ?? 0}`
      + ` · rollbacks ${health.rollbacks ?? 0} · last healthy batch ${health.last_healthy?.batch ?? '—'}`
    : '—';
  $('generatorStatus').textContent = heldout?.status === 'unavailable'
    ? `Held-out evaluation unavailable: ${heldout.reason || 'no reason recorded'}`
    : measured || free ? 'Inherited expert (the generator being trained).' : 'Waiting for the first generator measurement.';
  $('generatorDetail').textContent = [
    measured ? `Held-out: ${heldout.rows ?? '—'} windows · batch ${finite(heldout.batch) ? Number(heldout.batch).toLocaleString() : '—'}`
      + ` · checkpoint batch ${finite(heldout.checkpoint?.batch) ? Number(heldout.checkpoint.batch).toLocaleString() : '—'}`
      + ` · ${heldout.set_fingerprint || 'set fingerprint not recorded'}` : null,
    free ? `Free phase: ${free.rows ?? '—'} rows · batch ${finite(free.batch) ? Number(free.batch).toLocaleString() : '—'} · mask rate ${number(free.mask_rate, 2)}` : null,
  ].filter(Boolean).join(' · ');
  const lanes = state.focus_lanes?.lanes || {};
  const frozen = state.skip_request_expert_training === true;
  const laneText = Object.entries(lanes)
    .filter(([, lane]) => finite(lane?.heldout_accuracy))
    .map(([name, lane]) => `${name} ${percent(lane.heldout_accuracy)}`
      + (name === 'prose' && frozen ? ' (request expert (frozen in prose focus))' : ''));
  $('laneHeldout').textContent = laneText.length
    ? `Focus-lane held-out accuracy (request-conditioned expert${frozen ? ', not trained this run' : ''}): ${laneText.join(' · ')}`
    : '';
}

function renderCorpora(corpora) {
  const root = $('corpora');
  root.replaceChildren();
  for (const [name, count] of Object.entries(corpora)) {
    const row = document.createElement('div');
    row.className = 'corpus';
    const label = document.createElement('span');
    label.textContent = name.replaceAll('_', ' ');
    const value = document.createElement('strong');
    value.textContent = Number(count).toLocaleString();
    row.append(label, value);
    root.append(row);
  }
  const total = Object.values(corpora).reduce((sum, count) => sum + (Number(count) || 0), 0);
  $('corpusTotal').textContent = `${total.toLocaleString()} active examples this stage`;
}

function renderDatasetRegistry(datasets) {
  const root = $('datasetRegistry');
  root.replaceChildren();
  for (const dataset of datasets) {
    const row = document.createElement('div');
    row.className = `dataset ${dataset.status === 'active' ? 'active' : 'queued'}`;
    const copy = document.createElement('div');
    const name = dataset.source?.startsWith('https://huggingface.co/')
      ? document.createElement('a')
      : document.createElement('strong');
    name.textContent = dataset.id || 'unnamed dataset';
    if (name instanceof HTMLAnchorElement) {
      name.href = dataset.source;
      name.target = '_blank';
      name.rel = 'noopener noreferrer';
      name.title = 'Open the Hugging Face dataset page';
    }
    const kind = document.createElement('small');
    kind.textContent = dataset.kind || 'unknown';
    copy.append(name, kind);
    const state = document.createElement('span');
    const examples = Number(dataset.examples) || 0;
    const status = dataset.status === 'active'
      ? 'accepted for replay'
      : String(dataset.status).replaceAll('-', ' ');
    state.textContent = examples ? `${status} · ${examples.toLocaleString()}` : status;
    row.append(copy, state);
    root.append(row);
  }
}

function addHistory(event) {
  history.push(event);
  if (history.length > 240) history.shift();
  drawChart();
  renderEvents();
}

function drawChart() {
  const canvas = $('energyChart');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const pad = { left: 54, right: 18, top: 24, bottom: 34 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const values = history.flatMap(item => [Number(item.mean_free_energy ?? item.free_energy), Number(item.mean_positive_energy ?? item.positive_energy)]).filter(Number.isFinite);
  if (!values.length) {
    context.fillStyle = '#607169';
    context.font = '12px system-ui';
    context.fillText('Energy trace begins with the first settled batch.', pad.left, pad.top + 20);
    return;
  }
  let min = Math.min(...values), max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const margin = (max - min) * .08;
  min -= margin; max += margin;
  context.strokeStyle = 'rgba(164,207,185,.10)';
  context.fillStyle = '#607169';
  context.lineWidth = 1;
  context.font = '10px ui-monospace, monospace';
  for (let index = 0; index <= 4; index += 1) {
    const y = pad.top + plotHeight * index / 4;
    context.beginPath(); context.moveTo(pad.left, y); context.lineTo(width - pad.right, y); context.stroke();
    const label = max - (max - min) * index / 4;
    context.fillText(number(label, 2), 4, y + 3);
  }
  const draw = (field, fallback, color) => {
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    history.forEach((item, index) => {
      const value = Number(item[field] ?? item[fallback]);
      if (!Number.isFinite(value)) return;
      const x = pad.left + plotWidth * (history.length === 1 ? 1 : index / (history.length - 1));
      const y = pad.top + plotHeight * (max - value) / (max - min);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  };
  draw('mean_free_energy', 'free_energy', '#64d8ad');
  draw('mean_positive_energy', 'positive_energy', '#a690e8');
  context.fillStyle = '#607169';
  context.fillText(`${history.length} recent updates`, pad.left, height - 10);
}

function renderEvents() {
  const root = $('events');
  root.replaceChildren();
  for (const event of history.slice(-18).reverse()) {
    const row = document.createElement('div');
    row.className = 'event';
    const stamp = document.createElement('time');
    stamp.textContent = event.unix_millis ? new Date(Number(event.unix_millis)).toLocaleTimeString('en-US', { timeZone: 'America/Los_Angeles' }) : '—';
    const batch = document.createElement('strong');
    batch.textContent = `batch ${event.batch ?? '—'}`;
    const free = document.createElement('span');
    free.textContent = `free ${number(event.mean_free_energy ?? event.free_energy, 4)}`;
    const rate = document.createElement('span');
    rate.textContent = `${number(event.samples_per_second, 2)} samples/s`;
    row.append(stamp, batch, free, rate);
    root.append(row);
  }
}

function correlation(records, leftField, rightField) {
  const pairs = records
    .map(record => [record[leftField], record[rightField]])
    .filter(pair => pair.every(value => value !== null && value !== undefined && Number.isFinite(Number(value))))
    .map(pair => pair.map(Number));
  if (pairs.length < 3) return null;
  const leftMean = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
  const rightMean = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
  let numerator = 0, leftVariance = 0, rightVariance = 0;
  for (const [left, right] of pairs) {
    const leftDelta = left - leftMean;
    const rightDelta = right - rightMean;
    numerator += leftDelta * rightDelta;
    leftVariance += leftDelta * leftDelta;
    rightVariance += rightDelta * rightDelta;
  }
  const denominator = Math.sqrt(leftVariance * rightVariance);
  return denominator ? numerator / denominator : null;
}

function drawQualityChart() {
  const canvas = $('qualityChart');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const pad = { left: 48, right: 82, top: 28, bottom: 31 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  context.font = '10px ui-monospace, monospace';
  context.strokeStyle = 'rgba(164,207,185,.10)';
  context.fillStyle = '#607169';
  context.textAlign = 'left';
  for (let index = 0; index <= 4; index += 1) {
    const y = pad.top + plotHeight * index / 4;
    context.beginPath();
    context.moveTo(pad.left, y);
    context.lineTo(width - pad.right, y);
    context.stroke();
    context.fillText(`${100 - index * 25}%`, 6, y + 3);
  }
  context.fillText('quality', 6, 12);
  if (!auditHistory.length) {
    context.fillText('Jev audit history begins after the first graded sample batch.', pad.left, pad.top + 20);
    return;
  }
  const energies = auditHistory
    .flatMap(record => record.mean_free_energy === null || record.mean_free_energy === undefined
      ? []
      : [Number(record.mean_free_energy)])
    .filter(value => Number.isFinite(value) && value > 0);
  const energyMinimumValue = energies.length ? Math.min(...energies) : 1;
  const energyMaximumValue = energies.length ? Math.max(...energies) : 10;
  let energyMin = Math.log10(energyMinimumValue);
  let energyMax = Math.log10(energyMaximumValue);
  const energyPadding = Math.max((energyMax - energyMin) * .08, .02);
  energyMin -= energyPadding;
  energyMax += energyPadding;
  const point = (value, index, minimum = 0, maximum = 1) => ({
    x: pad.left + plotWidth * (auditHistory.length === 1 ? .5 : index / (auditHistory.length - 1)),
    y: pad.top + plotHeight * (maximum - value) / (maximum - minimum),
  });
  const draw = (
    field,
    color,
    minimum = 0,
    maximum = 1,
    dash = [],
    transform = value => value,
  ) => {
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 2;
    context.setLineDash(dash);
    context.beginPath();
    let started = false;
    let latest = null;
    auditHistory.forEach((record, index) => {
      const raw = record[field];
      if (raw === null || raw === undefined) return;
      const value = transform(Number(raw));
      if (!Number.isFinite(value)) return;
      latest = point(value, index, minimum, maximum);
      if (!started) {
        context.moveTo(latest.x, latest.y);
        started = true;
      } else {
        context.lineTo(latest.x, latest.y);
      }
    });
    context.stroke();
    context.setLineDash([]);
    if (latest) {
      context.beginPath();
      context.arc(latest.x, latest.y, 3, 0, Math.PI * 2);
      context.fill();
    }
  };
  draw('exact_accuracy', '#64d8ad');
  draw('jev_quality', '#a690e8');
  draw(
    'mean_free_energy',
    '#edb86b',
    energyMin,
    energyMax,
    [6, 5],
    value => value > 0 ? Math.log10(value) : Number.NaN,
  );
  context.fillStyle = '#607169';
  context.textAlign = 'right';
  context.fillText('free energy · log', width - 6, 12);
  context.fillText(number(10 ** energyMax, 0), width - 6, pad.top + 3);
  context.fillText(number(10 ** ((energyMin + energyMax) / 2), 0), width - 6, pad.top + plotHeight / 2 + 3);
  context.fillText(number(10 ** energyMin, 0), width - 6, pad.top + plotHeight + 3);
  context.textAlign = 'left';
  context.fillText(`${auditHistory.length} / 100 recent tests · latest points marked`, pad.left, height - 9);
}

function renderLatestAudit(audit, serverTime) {
  $('exactAccuracy').textContent = `${percent(audit.exact_accuracy)} · ${audit.exact_correct ?? '—'}/${audit.exact_total ?? '—'}`;
  $('jevQuality').textContent = percent(audit.jev_quality);
  $('contextFit').textContent = percent(audit.context_fit);
  $('languageQuality').textContent = percent(audit.language_quality);
  $('behavioralDiversity').textContent = percent(audit.behavioral_diversity);
  $('learnedSignal').textContent = percent(audit.learned_signal);
  $('auditStrength').textContent = String(audit.strongest_area || '—').replaceAll('_', ' ');
  $('auditFailure').textContent = String(audit.dominant_failure || '—').replaceAll('_', ' ');
  const age = Math.max(0, Math.round((Number(serverTime) - Number(audit.unix_millis)) / 1000));
  $('auditFreshness').textContent = `Jev ${audit.provider_model || ''} · ${audit.samples_graded || 0} samples · ${age}s ago`;
  const exactCorrelation = correlation(auditHistory, 'mean_free_energy', 'exact_accuracy');
  const qualityCorrelation = correlation(auditHistory, 'mean_free_energy', 'jev_quality');
  $('auditCorrelation').textContent = exactCorrelation == null || qualityCorrelation == null
    ? 'Correlation appears after three audits.'
    : `observed r(energy, strict pass) ${number(exactCorrelation, 3)} · r(energy, Jev quality) ${number(qualityCorrelation, 3)}`;
}
async function loadAudits() {
  try {
    const response = await fetch('/api/audits?limit=500', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const record = await response.json();
    const audits = Array.isArray(record.audits) ? record.audits : [];
    const currentAudits = audits.filter(audit =>
      audit.schema === 'river-jev-quality-metrics-v4'
      && Number(audit.samples_graded) >= 12);
    auditHistory = currentAudits.slice(-100);
    if (auditHistory.length) {
      renderLatestAudit(auditHistory.at(-1), record.server_unix_millis);
    } else {
      $('auditFreshness').textContent = 'Collecting the first complete 12-sample audit window.';
    }
    drawQualityChart();
  } catch (error) {
    $('auditFreshness').textContent = `Audit telemetry unavailable: ${error.message || error}`;
  }
}

async function loadHistory() {
  try {
    const response = await fetch('/api/events?limit=240', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const record = await response.json();
    for (const event of record.events || []) addHistory(event);
  } catch (error) {
    console.error('history load failed', error);
  }
}

function displayValue(value) {
  if (value && typeof value === 'object' && 'display' in value) {
    return `${value.display} · index ${value.index}`;
  }
  if (typeof value === 'string') return value;
  if (value === undefined) return 'Unavailable';
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function answerScope(scope) {
  if (scope === 'request_conditioned') return ['request-conditioned', 'Request-conditioned'];
  if (scope === 'inherited') return ['inherited', 'Foundation output'];
  return ['unavailable', 'Scope unavailable'];
}

function typedAnswer(name, answer, requested = {}) {
  const record = answer && typeof answer === 'object' ? answer : {};
  const type = record.type || requested.type || 'unknown';
  const card = document.createElement('article');
  card.className = 'typed-answer';
  const heading = document.createElement('div');
  heading.className = 'answer-heading';
  const title = document.createElement('strong');
  title.textContent = name;
  const badges = document.createElement('div');
  badges.className = 'answer-badges';
  const typeBadge = document.createElement('span');
  typeBadge.className = `type-badge ${type}`;
  typeBadge.textContent = type;
  const [scopeClass, scopeLabel] = answerScope(record.output_scope);
  const scopeBadge = document.createElement('span');
  scopeBadge.className = `scope-badge ${scopeClass}`;
  scopeBadge.textContent = scopeLabel;
  badges.append(typeBadge, scopeBadge);
  heading.append(title, badges);
  const body = document.createElement('pre');
  if (type === 'noul') {
    if (typeof record.noul === 'number' && Number.isFinite(record.noul)) {
      const criteria = requested.criteria || {};
      body.textContent = [
        `Noul activation ${number(record.noul, 7)}`,
        criteria.true ? `true · ${criteria.true}` : null,
        criteria.false ? `false · ${criteria.false}` : null,
      ].filter(Boolean).join('\n');
    } else {
      body.textContent = 'Noul value unavailable; the portal does not synthesize one.';
    }
  } else if (type === 'choice') {
    body.textContent = displayValue({
      choice: record.choice,
      probabilities: record.probabilities,
      confidence: record.confidence,
    });
  } else if (type === 'score') {
    body.textContent = displayValue({
      score: record.score,
      legend: record.legend,
      probabilities: record.probabilities,
      confidence: record.confidence,
    });
  } else if (type === 'text') {
    body.textContent = typeof record.text === 'string'
      ? record.text
      : 'Text value unavailable in telemetry.';
  } else if (type === 'structured') {
    body.textContent = Object.hasOwn(record, 'value')
      ? displayValue(record.value)
      : 'Structured value unavailable in telemetry.';
  } else {
    body.textContent = 'Unsupported answer type; raw value was not reinterpreted.';
  }
  card.append(heading, body);
  return card;
}

let visibleSampleRows = new Set();
let sampleListReady = false;

function compactSampleValue(value, limit = 180) {
  if (value === null || value === undefined) return 'Unavailable';
  const rendered = typeof value === 'string'
    ? value
    : typeof value === 'number'
      ? number(value, 5)
      : JSON.stringify(value);
  return rendered.length > limit ? `${rendered.slice(0, limit - 1)}…` : rendered;
}

function sampleSource(sample) {
  const source = sample.dataset_id
    || sample.supervision?.source
    || sample.request?.inputs?.source
    || sample.modality
    || sample.kind
    || 'unknown';
  if (source === 'legacy_pinball_v4' || source === 'pokemon_pinball_training') return 'Pokémon Pinball training set';
  return String(source).replaceAll('_', ' ');
}

function typedActual(record, type) {
  if (type === 'noul') return record?.noul;
  if (type === 'choice') {
    return {
      choice: record?.choice,
      probabilities: record?.probabilities,
      confidence: record?.confidence,
    };
  }
  if (type === 'score') {
    return {
      score: record?.score,
      legend: record?.legend,
      probabilities: record?.probabilities,
      confidence: record?.confidence,
    };
  }
  if (type === 'text') return record?.text;
  if (type === 'structured') return record?.value;
  return undefined;
}

function probabilityDiagnostics(probabilities, expected) {
  if (!probabilities || typeof probabilities !== 'object') return null;
  const target = finite(probabilities[expected]) ? Number(probabilities[expected]) : null;
  if (target === null) return null;
  const competitors = Object.entries(probabilities)
    .filter(([name, value]) => name !== String(expected) && finite(value))
    .map(([, value]) => Number(value));
  const strongest = competitors.length ? Math.max(...competitors) : 0;
  const margin = target - strongest;
  return {
    target,
    margin,
    tied: Math.abs(margin) <= 1e-6,
    uniqueTop: margin > 1e-6,
  };
}

function typedSampleRows(sample, sampleIndex) {
  const answers = sample.response?.answers;
  const requested = sample.request?.outputs;
  const names = new Set([
    ...Object.keys(requested && typeof requested === 'object' ? requested : {}),
    ...Object.keys(answers && typeof answers === 'object' ? answers : {}),
  ]);
  if (!names.size) names.add('answer unavailable');
  return Array.from(names, name => {
    const answer = answers?.[name];
    const request = requested?.[name] || {};
    const type = answer?.type || request.type || 'unknown';
    const actual = typedActual(answer, type);
    const supervised = sample.supervision?.label === name && finite(sample.supervision?.target);
    const target = supervised ? Number(sample.supervision.target) : null;
    const actualNumber = type === 'noul' && finite(actual) ? Number(actual) : null;
    const expectedText = supervised
      ? number(target, 5)
      : sample.kind === 'probe' ? 'Unsupervised probe' : 'No recorded target';
    const actualText = actualNumber !== null
      ? number(actualNumber, 7)
      : compactSampleValue(actual);
    let result = 'unscored';
    let resultClass = 'unscored';
    let absoluteError = null;
    let squaredError = null;
    let passed = null;
    let saturated = null;
    if (target !== null && actualNumber !== null) {
      const grade = gradeNoul(target, actualNumber);
      ({ absoluteError, squaredError, passed, saturated } = grade);
      resultClass = passed ? 'match' : 'miss';
      const reason = saturated
        ? 'saturated against soft target'
        : passed ? `within ±${NOUL_MAE_TOLERANCE}` : `outside ±${NOUL_MAE_TOLERANCE}`;
      result = `${passed ? 'pass' : 'fail'} · ${reason} · MAE ${number(absoluteError, 4)} · Brier ${number(squaredError, 4)}`;
    } else if (sample.response?.ok === false) {
      result = 'response failed';
      resultClass = 'miss';
    }
    const requestId = sample.request?.id || `${sampleIndex}`;
    return {
      key: `${requestId}:${name}`,
      kind: sample.kind || 'training',
      source: sampleSource(sample),
      task: name,
      detail: `${type} · ${answerScope(answer?.output_scope)[1]}`,
      expected: expectedText,
      actual: actualText,
      result,
      resultClass,
      typeGroup: type,
      absoluteError,
      squaredError,
      passed,
      saturated,
    };
  });
}

function foundationSampleRow(sample, sampleIndex) {
  const matched = sample.matched;
  const typeGroup = sample.output_type || sample.modality || 'foundation';
  let accuracy = finite(sample.accuracy)
    ? Number(sample.accuracy)
    : matched === true ? 1 : matched === false ? 0 : null;
  let resultClass = matched === true ? 'match' : matched === false ? 'miss' : 'unscored';
  let result = '';
  const metrics = {};

  if (typeGroup === 'choice' && typeof sample.expected === 'string') {
    const predictedChoice = sample.predicted?.choice;
    const diagnostic = probabilityDiagnostics(sample.predicted?.probabilities, sample.expected);
    metrics.labelCorrect = predictedChoice === sample.expected;
    metrics.uniqueTop = diagnostic?.uniqueTop ?? false;
    metrics.tie = diagnostic?.tied ?? false;
    accuracy = metrics.labelCorrect ? 1 : 0;
    resultClass = metrics.labelCorrect && metrics.uniqueTop
      ? 'match'
      : metrics.labelCorrect ? 'partial' : 'miss';
    const rank = diagnostic?.uniqueTop
      ? 'unique top'
      : diagnostic?.tied ? 'ranking tied' : 'target below top';
    result = `${metrics.labelCorrect ? 'label correct' : 'label incorrect'} · ${rank}`
      + (diagnostic ? ` · p(target) ${number(diagnostic.target, 4)} · margin ${number(diagnostic.margin, 4)}` : '');
  } else if (typeGroup === 'score' && finite(sample.expected?.score) && finite(sample.predicted?.score)) {
    metrics.absoluteError = Math.abs(Number(sample.expected.score) - Number(sample.predicted.score));
    const expectedLevel = String(sample.expected.score);
    const diagnostic = probabilityDiagnostics(sample.predicted?.probabilities, expectedLevel);
    metrics.uniqueTop = diagnostic?.uniqueTop ?? false;
    metrics.tie = diagnostic?.tied ?? false;
    const pointExact = metrics.absoluteError <= 1e-6;
    resultClass = pointExact && metrics.uniqueTop ? 'match' : pointExact ? 'partial' : 'miss';
    accuracy = null;
    const point = pointExact ? 'point exact' : `point Δ ${number(metrics.absoluteError, 4)}`;
    const distribution = diagnostic?.uniqueTop
      ? 'distribution unique top'
      : diagnostic?.tied ? 'distribution tied' : 'target below top';
    result = `${point} · ${distribution}`
      + (diagnostic ? ` · p(target) ${number(diagnostic.target, 4)}` : '');
  } else {
    if (typeGroup === 'structured') {
      metrics.exact = matched === true;
      metrics.fieldAccuracy = accuracy;
      resultClass = matched === true ? 'match' : accuracy > 0 ? 'partial' : 'miss';
    } else if (['prose', 'code'].includes(typeGroup) && sample.aggregate !== true) {
      metrics.exact = matched === true;
    }
    const resultParts = [
      resultClass,
      accuracy === null ? null : percent(accuracy),
      sample.schema_valid === false ? 'invalid schema' : null,
      sample.diff,
    ].filter(Boolean);
    result = resultParts.join(' · ');
  }
  return {
    key: `${sample.epoch ?? 'unknown'}:${sample.batch ?? sampleIndex}:${typeGroup}`,
    kind: sample.kind || 'evaluation',
    source: sampleSource(sample),
    task: sample.task || `${typeGroup} output`,
    detail: `epoch ${sample.epoch ?? '—'} · batch ${sample.batch ?? '—'}`,
    expected: compactSampleValue(sample.expected),
    actual: compactSampleValue(sample.predicted),
    result,
    resultClass,
    typeGroup,
    accuracy,
    ...metrics,
  };
}

function sampleCell(className, label, value) {
  const cell = document.createElement('div');
  cell.className = className;
  cell.dataset.label = label;
  cell.textContent = value;
  cell.title = value;
  return cell;
}

function sampleAccuracySummary(rows) {
  const summary = document.createElement('div');
  summary.className = 'sample-accuracy-summary';
  const average = (items, field) => {
    const values = items.map(item => item[field]).filter(finite).map(Number);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  for (const type of SAMPLE_TYPES) {
    const scored = rows.filter(row => row.typeGroup === type);
    const item = document.createElement('div');
    const label = document.createElement('span');
    const value = document.createElement('strong');
    label.textContent = type;
    if (type === 'noul') {
      const usable = scored.filter(row => finite(row.absoluteError));
      const passes = usable.filter(row => row.passed).length / Math.max(1, usable.length);
      const saturations = usable.filter(row => row.saturated).length / Math.max(1, usable.length);
      value.textContent = usable.length
        ? `pass ${percent(passes)} · saturated ${percent(saturations)} · MAE ${number(average(usable, 'absoluteError'), 3)} · ${usable.length}`
        : 'waiting';
    } else if (type === 'choice') {
      const usable = scored.filter(row => typeof row.labelCorrect === 'boolean');
      const labels = usable.filter(row => row.labelCorrect).length / Math.max(1, usable.length);
      const unique = usable.filter(row => row.labelCorrect && row.uniqueTop).length / Math.max(1, usable.length);
      const ties = usable.filter(row => row.tie).length;
      value.textContent = usable.length
        ? `label ${percent(labels)} · top ${percent(unique)} · ${ties} tie${ties === 1 ? '' : 's'}`
        : 'waiting';
    } else if (type === 'score') {
      const usable = scored.filter(row => finite(row.absoluteError));
      const unique = usable.filter(row => row.uniqueTop).length / Math.max(1, usable.length);
      value.textContent = usable.length
        ? `MAE ${number(average(usable, 'absoluteError'), 3)} · top ${percent(unique)} · ${usable.length}`
        : 'waiting';
    } else if (type === 'structured') {
      const usable = scored.filter(row => typeof row.exact === 'boolean');
      const exact = usable.filter(row => row.exact).length / Math.max(1, usable.length);
      value.textContent = usable.length
        ? `exact ${percent(exact)} · fields ${percent(average(usable, 'fieldAccuracy'))} · ${usable.length}`
        : 'waiting';
    } else {
      const usable = scored.filter(row => typeof row.exact === 'boolean');
      const exact = usable.filter(row => row.exact).length / Math.max(1, usable.length);
      value.textContent = usable.length ? `exact ${percent(exact)} · ${usable.length}` : 'waiting';
    }
    item.append(label, value);
    summary.append(item);
  }
  return summary;
}

function renderSamples(samples) {
  const root = $('trainingSamples');
  root.replaceChildren();
  // samples.jsonl survives restarts; rows numbered past the live batch belong to an earlier run.
  const liveBatch = Number(lastState?.batch);
  // Only prose questions with generated text: the input the model saw and the text it wrote.
  const rows = samples
    .filter(sample => !(finite(sample.batch) && finite(liveBatch) && Number(sample.batch) > liveBatch))
    .filter(isProseTextSample)
    .slice()
    .reverse()
    .map((sample, index) => ({
      ...foundationSampleRow(sample, index),
      input: proseInputText(sample),
      // Newlines and tabs stay visible as ↵ and ⇥ so a run of them reads as output, not blank space.
      actual: sample.predicted.replace(/\n/g, '↵').replace(/\t/g, '⇥') || '(empty)',
    }))
    .slice(0, SAMPLE_ROWS);
  if (!rows.length) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'Waiting for the first prose answer from this run.';
    root.append(empty);
    visibleSampleRows = new Set();
    sampleListReady = false;
    return 0;
  }

  const nextVisibleRows = new Set(rows.map(row => row.key));

  root.append(sampleAccuracySummary(rows));
  const heading = document.createElement('div');
  heading.className = 'sample-list-heading';
  for (const label of ['Input', 'Expected', 'Output', 'Result']) {
    const cell = document.createElement('span');
    cell.textContent = label;
    heading.append(cell);
  }
  root.append(heading);

  for (const row of rows) {
    const item = document.createElement('article');
    const isNew = sampleListReady && !visibleSampleRows.has(row.key);
    item.className = `sample-row ${row.resultClass}${isNew ? ' is-new' : ''}`;

    const identity = document.createElement('div');
    identity.className = 'sample-identity';
    const topLine = document.createElement('div');
    topLine.className = 'sample-identity-top';
    const task = document.createElement('strong');
    task.textContent = row.input;
    topLine.append(task);
    if (isNew) {
      const marker = document.createElement('span');
      marker.className = 'new-sample-marker';
      marker.textContent = 'new';
      topLine.append(marker);
    }
    const context = document.createElement('small');
    context.textContent = `${row.task} · ${row.detail}`;
    identity.append(topLine, context);

    item.append(
      identity,
      sampleCell('sample-value sample-expected', 'Expected', row.expected),
      sampleCell('sample-value sample-actual', 'Output', row.actual),
    );
    const result = sampleCell('sample-result', 'Result', row.result);
    result.classList.add(row.resultClass);
    item.append(result);
    root.append(item);
  }

  visibleSampleRows = nextVisibleRows;
  sampleListReady = true;
  return rows.length;
}

// The last held-out evaluations of this run, so a change in learning shows as a moving series.
function renderHeldoutTrend(samples) {
  const liveBatch = Number(lastState?.batch);
  const seen = new Set();
  const points = samples
    .filter(sample => sample.dataset_id === 'generator-heldout-v1' && finite(sample.top1_accuracy)
      && !(finite(liveBatch) && Number(sample.batch) > liveBatch))
    .filter(sample => !seen.has(sample.batch) && seen.add(sample.batch))
    .slice(-8);
  $('generatorTrend').textContent = points.length
    ? points.map(sample => `b${sample.batch} ${percent(sample.top1_accuracy)}`).join(' → ')
    : '—';
}

async function loadSamples(batch = '') {
  if (samplesLoading) return;
  samplesLoading = true;
  try {
    const response = await fetch(`/api/samples?limit=${SAMPLE_FETCH_LIMIT}`, {
      cache: 'no-store',
    });
    const body = await response.text();
    let record;
    try {
      record = JSON.parse(body);
    } catch {
      throw new Error(`HTTP ${response.status}: ${body.slice(0, 120) || 'non-JSON response'}`);
    }
    if (!response.ok) throw new Error(record.error || `HTTP ${response.status}`);
    const shown = renderSamples(record.samples || []);
    renderHeldoutTrend(record.samples || []);
    $('sampleStatus').textContent = `${shown} recent prose answers · newest first · a new question about every 15 minutes`;
    lastSamplesBatch = String(batch);
  } catch (error) {
    $('sampleStatus').textContent = String(error.message || error);
  } finally {
    samplesLoading = false;
  }
}

function connectStream() {
  const stream = new EventSource('/api/stream');
  stream.addEventListener('state', event => {
    try { setState(JSON.parse(event.data)); } catch (error) { console.error(error); }
  });
  stream.onopen = () => { $('connectionDot').classList.add('live'); };
  stream.onerror = () => {
    $('connectionDot').className = 'dot error';
    $('freshness').textContent = 'reconnecting';
  };
}

const defaultChoiceCriteria = {
  refund: 'The customer asks for money to be returned.',
  information: 'The customer asks only for information.',
};
const defaultScoreCriteria = [
  'Low or absent.',
  'Present but moderate.',
  'Strong or explicit.',
];

const defaultSchema = {
  type: 'object',
  properties: {
    answer: { type: 'string', max_length: 96 },
    confidence: { type: 'number', minimum: 0, maximum: 1 },
  },
  required: ['answer', 'confidence'],
};
let outputRequestCount = 0;

function labelledField(text, control, className = '') {
  const label = document.createElement('label');
  label.className = className;
  label.append(document.createTextNode(text), control);
  return label;
}

function outputControl(role, kind = 'input') {
  const control = document.createElement(kind);
  control.dataset.role = role;
  return control;
}

function refreshOutputCard(card) {
  const type = card.querySelector('[data-role="type"]').value;
  const typed = ['noul', 'choice', 'score'].includes(type);
  card.querySelector('.criteria-true-field').hidden = type !== 'noul';
  card.querySelector('.criteria-false-field').hidden = type !== 'noul';
  card.querySelector('.criteria-options-field').hidden = !['choice', 'score'].includes(type);
  card.querySelector('.max-bytes-field').hidden = typed;
  card.querySelector('.schema-field').hidden = type !== 'structured';
  const criteria = card.querySelector('[data-role="criteria_options"]');
  if (type === 'choice' || type === 'score') {
    let parsed;
    try { parsed = JSON.parse(criteria.value); } catch { parsed = null; }
    const validShape = type === 'choice'
      ? parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      : Array.isArray(parsed);
    if (!validShape) {
      criteria.value = JSON.stringify(
        type === 'choice' ? defaultChoiceCriteria : defaultScoreCriteria,
        null,
        2,
      );
    }
  }
  for (const remove of document.querySelectorAll('.remove-output')) {
    remove.disabled = document.querySelectorAll('.output-request').length === 1;
  }
  $('addOutput').disabled = document.querySelectorAll('.output-request').length >= 64;
}

function addOutput(initial = {}) {
  if (document.querySelectorAll('.output-request').length >= 64) return;
  outputRequestCount += 1;
  const card = document.createElement('article');
  card.className = 'output-request';
  const head = document.createElement('div');
  head.className = 'output-request-head';
  const name = outputControl('name');
  name.value = initial.name || (outputRequestCount === 1 ? 'answer' : `output_${outputRequestCount}`);
  name.required = true;
  name.maxLength = 128;
  const type = outputControl('type', 'select');
  for (const value of ['noul', 'choice', 'score', 'text', 'structured']) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value === 'noul' ? 'Noul' : value[0].toUpperCase() + value.slice(1);
    type.append(option);
  }
  type.value = initial.type || 'structured';
  const remove = document.createElement('button');
  remove.className = 'remove-output';
  remove.type = 'button';
  remove.title = 'Remove output';
  remove.setAttribute('aria-label', 'Remove output');
  remove.textContent = '×';
  remove.addEventListener('click', () => {
    card.remove();
    for (const item of document.querySelectorAll('.output-request')) refreshOutputCard(item);
  });
  head.append(labelledField('Output name', name), labelledField('Answer type', type), remove);

  const fields = document.createElement('div');
  fields.className = 'output-fields';
  const instructions = outputControl('instructions', 'textarea');
  instructions.rows = 3;
  instructions.required = true;
  instructions.value = initial.instructions || 'Answer this request using the supplied input.';
  const maxBytes = outputControl('max_bytes');
  maxBytes.type = 'number';
  maxBytes.min = '1';
  maxBytes.max = '65536';
  maxBytes.value = String(initial.max_bytes || 128);
  const criteriaTrue = outputControl('criteria_true');
  criteriaTrue.value = initial.criteria?.true || 'The requested condition is satisfied.';
  const criteriaFalse = outputControl('criteria_false');
  criteriaFalse.value = initial.criteria?.false || 'The requested condition is not satisfied.';
  const criteriaOptions = outputControl('criteria_options', 'textarea');
  criteriaOptions.rows = 7;
  criteriaOptions.spellcheck = false;
  criteriaOptions.value = JSON.stringify(
    initial.criteria || (initial.type === 'score' ? defaultScoreCriteria : defaultChoiceCriteria),
    null,
    2,
  );
  const schema = outputControl('schema', 'textarea');
  schema.rows = 9;
  schema.spellcheck = false;
  schema.value = JSON.stringify(initial.schema || defaultSchema, null, 2);
  fields.append(
    labelledField('Instructions', instructions, 'instructions-field'),
    labelledField('Max bytes', maxBytes, 'max-bytes-field'),
    labelledField('True criterion', criteriaTrue, 'criteria-true-field'),
    labelledField('False criterion', criteriaFalse, 'criteria-false-field'),
    labelledField('Options or ordered levels (JSON)', criteriaOptions, 'criteria-options-field'),
    labelledField('JSON schema', schema, 'schema-field'),
  );
  card.append(head, fields);
  type.addEventListener('change', () => refreshOutputCard(card));
  $('outputRequests').append(card);
  refreshOutputCard(card);
}

function collectOutputs() {
  const outputs = Object.create(null);
  for (const card of document.querySelectorAll('.output-request')) {
    const name = card.querySelector('[data-role="name"]').value.trim();
    if (!name) throw new Error('Every output needs a name.');
    if (Object.hasOwn(outputs, name)) throw new Error(`Output name "${name}" is duplicated.`);
    const type = card.querySelector('[data-role="type"]').value;
    const instructions = card.querySelector('[data-role="instructions"]').value;
    if (!instructions.trim()) throw new Error(`Output "${name}" needs instructions.`);
    if (type === 'noul') {
      outputs[name] = {
        type,
        instructions,
        criteria: {
          true: card.querySelector('[data-role="criteria_true"]').value,
          false: card.querySelector('[data-role="criteria_false"]').value,
        },
      };
    } else if (type === 'choice' || type === 'score') {
      outputs[name] = {
        type,
        instructions,
        criteria: JSON.parse(card.querySelector('[data-role="criteria_options"]').value),
      };
    } else {
      const maxBytes = Number(card.querySelector('[data-role="max_bytes"]').value);
      outputs[name] = { type, instructions, max_bytes: maxBytes };
      if (type === 'structured') {
        outputs[name].schema = JSON.parse(card.querySelector('[data-role="schema"]').value);
      }
    }
  }
  return outputs;
}

function renderProbeAnswers(record, request) {
  const root = $('probeOutput');
  root.replaceChildren();
  if (!record.answers || typeof record.answers !== 'object' || Array.isArray(record.answers)) {
    throw new Error('Runtime response did not contain a named answers map.');
  }
  const entries = Object.entries(record.answers);
  if (!entries.length) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'The runtime returned no answers.';
    root.append(empty);
    return;
  }
  for (const [name, answer] of entries) {
    root.append(typedAnswer(name, answer, request.outputs[name]));
  }
}

$('addOutput').addEventListener('click', () => addOutput({ type: 'text', instructions: 'Write a concise plain-text answer.' }));
localStorage.removeItem('riverPortalToken');
addOutput();

$('probeForm').addEventListener('submit', async event => {
  event.preventDefault();
  const button = $('runProbe');
  button.disabled = true;
  $('probeState').textContent = 'Queued';
  $('probeOutput').replaceChildren(Object.assign(document.createElement('p'), {
    className: 'empty',
    textContent: 'Waiting for the trainer to finish its current batch…',
  }));
  try {
    const payload = {
      id: globalThis.crypto?.randomUUID?.() || `portal-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      inputs: {
        prompt: $('prompt').value,
        modality: $('modality').value,
      },
      outputs: collectOutputs(),
    };
    const response = await fetch('/api/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const responseBody = await response.text();
    let record;
    try {
      record = JSON.parse(responseBody);
    } catch {
      throw new Error(`HTTP ${response.status}: ${responseBody.slice(0, 160) || 'non-JSON response'}`);
    }
    if (!response.ok || !record.ok) throw new Error(record.error || `HTTP ${response.status}`);
    renderProbeAnswers(record, payload);
    $('probeState').textContent = 'Complete';
  } catch (error) {
    $('probeState').textContent = 'Failed';
    $('probeOutput').replaceChildren(Object.assign(document.createElement('p'), {
      className: 'empty',
      textContent: String(error.message || error),
    }));
  } finally {
    button.disabled = false;
  }
});

window.addEventListener('resize', () => {
  drawChart();
  drawQualityChart();
});
setInterval(() => { $('clock').textContent = pacific(Date.now()); }, 1000);
setInterval(loadAudits, 10_000);
setInterval(loadFridayUpdate, 5_000);
setInterval(loadCapabilities, 30_000);
loadCapabilities();
loadFridayUpdate();
loadHistory();
loadAudits();
connectStream();
