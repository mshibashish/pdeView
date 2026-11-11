export const fmtSci = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toExponential(digits).replace("e", "e");

export const fmtNum = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(digits);

/** Signed value with a true minus sign. */
export const fmtSigned = (v: number, digits = 4): string =>
  `${v < 0 ? "−" : v > 0 ? "+" : ""}${Math.abs(v).toFixed(digits)}`;

export const metricLabel: Record<string, string> = {
  RMSE: "RMSE",
  nRMSE: "nRMSE",
  cRMSE: "cRMSE (conserved)",
  max_error: "Max error",
  bRMSE: "bRMSE (boundary)",
  fRMSE_low: "fRMSE low k",
  fRMSE_mid: "fRMSE mid k",
  fRMSE_high: "fRMSE high k",
};
