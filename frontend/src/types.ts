export type MetricKey =
  | "RMSE"
  | "nRMSE"
  | "cRMSE"
  | "max_error"
  | "bRMSE"
  | "fRMSE_low"
  | "fRMSE_mid"
  | "fRMSE_high";

export type Metrics = Record<MetricKey, number>;

export interface Criterion {
  name: string;
  pass: boolean;
  kind: "gate" | "finding";
}

export interface Meta {
  grid: { x: number[]; y: number[]; domain: [number, number, number, number] };
  t: number[];
  shape: [number, number, number];
  initial_step: number;
  physics: Record<string, unknown>;
  model: {
    name: string;
    device: string;
    torch: string;
    n_parameters: number;
    epoch: number;
    md5_ok: boolean;
    modes: number;
    width: number;
    initial_step: number;
  };
  verification: {
    available: boolean;
    verdict: "PASS" | "FAIL" | null;
    generated_at: string | null;
    criteria: Criterion[];
    paper: Metrics | null;
    pdebench_compat: Metrics | null;
    clean: Metrics | null;
    persistence_over_fno_nRMSE: number | null;
  };
  dataset: {
    n_samples: number | null;
    n_distinct_trajectories: number | null;
    n_possible_initial_conditions: number | null;
    test_with_identical_train_trajectory: number | null;
    n_test: number | null;
  };
}

export interface SampleSummary {
  key: string;
  split: "test" | "train";
  dam_radius: number;
  nRMSE: number | null;
  RMSE: number | null;
  max_error: number | null;
  ic_cells: number | null;
  n_twins_total: number;
  twins_local: string[];
}

export interface SampleStats {
  key: string;
  t: number[];
  initial_step: number;
  rmse: (number | null)[];
  nrmse: (number | null)[];
  max_abs: (number | null)[];
  persistence_nrmse: (number | null)[];
  volume_gt: number[];
  volume_pred: number[];
  range: { min: number; max: number; err_abs_max: number };
  summary: { nRMSE: number; RMSE: number; max_error: number; persistence_nRMSE: number };
}

/** A [T, X, Y] float32 field stored row-major (x-major, like the backend). */
export interface Field {
  data: Float32Array;
  nt: number;
  nx: number;
  ny: number;
}

export interface SampleData {
  key: string;
  gt: Field;
  pred: Field;
  stats: SampleStats;
}

export interface Verification {
  verdict: string;
  generated_at: string;
  criteria: {
    name: string;
    value: unknown;
    threshold: string;
    pass: boolean;
    detail: string;
    kind: "gate" | "finding";
  }[];
  checkpoint: Record<string, unknown>;
  fingerprint: {
    stored_val_loss: number;
    reproduced_val_loss: number;
    relative_difference: number;
    horizon_sweep: Record<string, number>;
    grid_ablation_mse: Record<string, number>;
  };
  audit: {
    n_samples: number;
    n_possible_initial_conditions: number;
    n_distinct_trajectories: number;
    n_distinct_train: number;
    n_distinct_test: number;
    test_with_identical_train_trajectory: number;
    n_test: number;
    copies_per_trajectory: { min: number; median: number; max: number };
  };
  accuracy: {
    paper: Metrics;
    pdebench_compat: Metrics;
    clean: Metrics;
    ratio_to_paper: Metrics;
    persistence_clean: Metrics;
    train_probe_clean: Metrics;
    dedup_test_clean: Metrics;
    persistence_over_fno_nRMSE: number;
    n_dedup_test: number;
  };
  device: Record<string, number | boolean | null>;
  environment: Record<string, string | number>;
}
