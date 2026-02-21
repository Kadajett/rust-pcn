/** JSONL event types emitted by pcn-train */

export interface EpochEvent {
  type: "epoch";
  epoch: number;
  avg_energy: number;
  accuracy: number;
  layer_errors: number[];
  elapsed_secs: number;
  num_samples: number;
  num_books: number;
}

export interface EvalEvent {
  type: "eval";
  epoch: number;
  book: string;
  accuracy: number;
  sample_predictions: SamplePrediction[];
}

export interface SamplePrediction {
  input: string;
  predicted: string;
  expected: string;
  correct: boolean;
}

export interface NewBookEvent {
  type: "new_book";
  epoch: number;
  book: string;
  samples: number;
  train_samples: number;
  eval_samples: number;
}

export interface CheckpointEvent {
  type: "checkpoint";
  epoch: number;
  path: string;
}

export type TrainingEvent = EpochEvent | EvalEvent | NewBookEvent | CheckpointEvent;
