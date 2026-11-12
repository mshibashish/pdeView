import type { DamParams, SampleStats } from "../types";

export interface SimRequest {
  id: number;
  key: string;
  params: DamParams;
  t: number[];
  initialStep: number;
  modelUrl: string;
}

export type SimProgress =
  | { stage: "solver" }
  | { stage: "model" }
  | { stage: "rollout"; step: number; total: number };

export type SimResponse =
  | ({ id: number; type: "progress" } & SimProgress)
  | {
      id: number; type: "done"; gt: Float32Array; pred: Float32Array; stats: SampleStats;
      solverSeconds: number; rolloutSeconds: number;
    }
  | { id: number; type: "error"; message: string };
