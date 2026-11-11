/**
 * Colour scales for fields, from the reference data-viz palette.
 *
 * - Water height (magnitude): one hue, the sequential blue ramp. In light mode low = light,
 *   high = dark; in dark mode the anchor flips so low values recede into the dark surface.
 * - Signed error (polarity): diverging blue <-> red with a neutral gray midpoint at zero.
 */

export type Theme = "light" | "dark";

const SEQ_BLUE = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
  "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
];

const DIVERGING: Record<Theme, string[]> = {
  light: ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#a32b2a"],
  dark: ["#86b6ef", "#3987e5", "#1c5cab", "#383835", "#a83c3b", "#e66767", "#f3b0b0"],
};

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** sRGB <-> linear helpers so interpolation between stops does not muddy the midtones. */
const toLin = (c: number) => {
  const s = c / 255;
  return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
};
const toSrgb = (l: number) => {
  const s = l <= 0.0031308 ? 12.92 * l : 1.055 * l ** (1 / 2.4) - 0.055;
  return Math.round(Math.min(1, Math.max(0, s)) * 255);
};

/** 256-entry RGBA lookup table interpolated through the given stops. */
function buildLut(stops: string[]): Uint8ClampedArray {
  const rgb = stops.map(hexToRgb).map((c) => c.map(toLin) as [number, number, number]);
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const pos = (i / 255) * (rgb.length - 1);
    const j = Math.min(rgb.length - 2, Math.floor(pos));
    const f = pos - j;
    for (let c = 0; c < 3; c++) {
      lut[i * 4 + c] = toSrgb(rgb[j][c] * (1 - f) + rgb[j + 1][c] * f);
    }
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

const cache = new Map<string, Uint8ClampedArray>();

export function lut(kind: "height" | "error", theme: Theme): Uint8ClampedArray {
  const id = `${kind}-${theme}`;
  let l = cache.get(id);
  if (!l) {
    const stops =
      kind === "error"
        ? DIVERGING[theme]
        : theme === "light"
          ? SEQ_BLUE
          : [...SEQ_BLUE].reverse();
    l = buildLut(stops);
    cache.set(id, l);
  }
  return l;
}

/** CSS linear-gradient for a colour bar. */
export function gradientCss(kind: "height" | "error", theme: Theme): string {
  const l = lut(kind, theme);
  const stops: string[] = [];
  for (let i = 0; i <= 16; i++) {
    const k = Math.round((i / 16) * 255) * 4;
    stops.push(`rgb(${l[k]}, ${l[k + 1]}, ${l[k + 2]}) ${(i / 16) * 100}%`);
  }
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

/**
 * Paint one [X, Y] frame into an RGBA buffer laid out for a canvas with x to the right and
 * y upwards (row 0 = largest y).
 */
export function paintFrame(
  out: Uint8ClampedArray,
  values: Float32Array | ((ix: number, iy: number) => number),
  nx: number,
  ny: number,
  lo: number,
  hi: number,
  table: Uint8ClampedArray,
): void {
  const scale = 255 / (hi - lo || 1);
  const get = typeof values === "function" ? values : (ix: number, iy: number) => values[ix * ny + iy];
  for (let iy = 0; iy < ny; iy++) {
    const row = ny - 1 - iy;
    for (let ix = 0; ix < nx; ix++) {
      let k = Math.round((get(ix, iy) - lo) * scale);
      k = k < 0 ? 0 : k > 255 ? 255 : k;
      const o = (row * nx + ix) * 4;
      out[o] = table[k * 4];
      out[o + 1] = table[k * 4 + 1];
      out[o + 2] = table[k * 4 + 2];
      out[o + 3] = 255;
    }
  }
}
