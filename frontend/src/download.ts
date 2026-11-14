/** Save a blob under `name` through a temporary object URL. */
export function downloadBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

const cell = (v: string | number | null | undefined): string => {
  if (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v))) return "";
  const s = String(v);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** UTF-8 byte-order mark, so spreadsheets read labels such as "ΔV" correctly. */
const BOM = String.fromCharCode(0xfeff);

/** RFC 4180 CSV (CRLF line ends, quoted fields where needed). */
export function csvBlob(header: string[], rows: (string | number | null | undefined)[][]): Blob {
  const lines = [header, ...rows].map((r) => r.map(cell).join(","));
  return new Blob([BOM, lines.join("\r\n"), "\r\n"], { type: "text/csv;charset=utf-8" });
}
