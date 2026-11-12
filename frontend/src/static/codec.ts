// Decoder for the static site's field files (format: docs/static-site.md, "Field encoding").
import type { Field } from "../types";

const HEADER = 40;
const MAGIC = "PDVF";

async function gunzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

/** Decode a `.bin` field: gzip(header ‖ byte-shuffled payload) -> float32 [T, X, Y]. */
export async function decodeField(raw: ArrayBuffer, label: string): Promise<Field> {
  const head = new Uint8Array(raw, 0, Math.min(2, raw.byteLength));
  // Our own gzip layer is normally still there; a proxy that strips it is tolerated.
  const buf = head[0] === 0x1f && head[1] === 0x8b ? await gunzip(raw) : raw;
  if (buf.byteLength < HEADER) throw new Error(`${label}: truncated field file`);
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== MAGIC) throw new Error(`${label}: not a pdeView field file`);
  const version = dv.getUint8(4);
  const codec = dv.getUint8(5);
  if (version !== 1) throw new Error(`${label}: unsupported field version ${version}`);
  const nt = dv.getUint32(8, true);
  const nx = dv.getUint32(12, true);
  const ny = dv.getUint32(16, true);
  const lo = dv.getFloat64(24, true);
  const scale = dv.getFloat64(32, true);
  const n = nt * nx * ny;
  const payload = new Uint8Array(buf, HEADER);

  if (codec === 1) {
    if (payload.length !== 4 * n) throw new Error(`${label}: size mismatch`);
    const bytes = new Uint8Array(4 * n);
    for (let b = 0; b < 4; b++) {
      const plane = payload.subarray(b * n, (b + 1) * n);
      for (let i = 0; i < n; i++) bytes[4 * i + b] = plane[i];
    }
    return { data: new Float32Array(bytes.buffer), nt, nx, ny };
  }
  if (codec === 2) {
    if (payload.length !== 2 * n) throw new Error(`${label}: size mismatch`);
    const lowB = payload.subarray(0, n);
    const highB = payload.subarray(n, 2 * n);
    const s = nx * ny;
    const q = new Uint16Array(n);
    for (let i = 0; i < s; i++) q[i] = lowB[i] | (highB[i] << 8);
    for (let i = s; i < n; i++) q[i] = (q[i - s] + (lowB[i] | (highB[i] << 8))) & 0xffff;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = lo + q[i] * scale;
    return { data, nt, nx, ny };
  }
  throw new Error(`${label}: unknown codec ${codec}`);
}
