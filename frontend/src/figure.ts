// Export figure: the current view drawn onto one canvas, with its colour bars and a caption, so a saved
// PNG can be read on its own. Colours come from the page's theme tokens, so it matches the screen.
import { paintFrame } from "./colormap";

export interface FigurePanel {
  title: string;
  subtitle?: string;
  /** A [X, Y] frame (x-major) or an accessor; or an image that is already rendered (the 3D view). */
  values?: Float32Array | ((ix: number, iy: number) => number);
  image?: CanvasImageSource & { width: number; height: number };
  lo?: number;
  hi?: number;
  table?: Uint8ClampedArray;
}

export interface FigureBar {
  table: Uint8ClampedArray;
  lo: number;
  hi: number;
  label: string;
  format: (v: number) => string;
}

export interface FigureSpec {
  title: string;
  subtitle: string;
  panels: FigurePanel[];
  bars: FigureBar[];
  footer: string;
  nx: number;
  ny: number;
  /** Width of one panel in CSS pixels. */
  panelWidth: number;
}

const SCALE = 2; // device pixels per CSS pixel in the PNG
const M = 28; // outer margin
const GAP = 20;
const BAR_W = 300;

function tokens() {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return {
    bg: v("--surface-1"), text: v("--text-primary"), secondary: v("--text-secondary"), muted: v("--text-muted"),
    border: v("--border"), font: v("--font") || "system-ui, sans-serif",
  };
}

/** Render the figure; the caller turns it into a PNG. */
export function renderFigure(spec: FigureSpec): HTMLCanvasElement {
  const t = tokens();
  const { panels, bars, nx, ny, panelWidth: P } = spec;
  const panelH = (p: FigurePanel) => (p.image ? Math.round((P * p.image.height) / p.image.width) : P);
  const imgH = Math.max(...panels.map(panelH));
  const width = M * 2 + panels.length * P + (panels.length - 1) * GAP;
  const panelsTop = M + 50;
  const barsTop = panelsTop + 24 + imgH + 26;
  const footerY = barsTop + 80;
  const height = footerY + M - 10;

  const canvas = document.createElement("canvas");
  canvas.width = width * SCALE;
  canvas.height = height * SCALE;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2D context unavailable");
  ctx.scale(SCALE, SCALE);
  ctx.fillStyle = t.bg;
  ctx.fillRect(0, 0, width, height);
  ctx.textBaseline = "alphabetic";
  const text = (s: string, x: number, y: number, size: number, color: string, weight = 400, align: CanvasTextAlign = "left") => {
    ctx.font = `${weight} ${size}px ${t.font}`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.fillText(s, x, y);
  };

  text(spec.title, M, M + 14, 17, t.text, 600);
  text(spec.subtitle, M, M + 34, 12.5, t.secondary);

  // panels: title row, then the field upscaled with nearest-neighbour sampling (cells stay square)
  const cells = document.createElement("canvas");
  cells.width = nx;
  cells.height = ny;
  const cctx = cells.getContext("2d");
  const image = cctx?.createImageData(nx, ny);
  panels.forEach((p, i) => {
    const x = M + i * (P + GAP);
    text(p.title, x, panelsTop + 12, 13, t.text, 600);
    if (p.subtitle) text(p.subtitle, x + P, panelsTop + 12, 11.5, t.muted, 400, "right");
    const y = panelsTop + 24;
    ctx.imageSmoothingEnabled = false;
    if (p.image) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(p.image, x, y, P, panelH(p));
    } else if (p.values && p.table && p.lo !== undefined && p.hi !== undefined && cctx && image) {
      paintFrame(image.data, p.values, nx, ny, p.lo, p.hi, p.table);
      cctx.putImageData(image, 0, 0);
      ctx.drawImage(cells, x, y, P, P);
    }
    ctx.strokeStyle = t.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, P - 1, panelH(p) - 1);
  });

  // colour bars, as on screen: label, ramp, and min / mid / max ticks
  bars.forEach((b, i) => {
    const x = M + i * (BAR_W + 40);
    text(b.label, x, barsTop + 10, 12, t.secondary);
    for (let px = 0; px < BAR_W; px++) {
      const k = Math.round((px / (BAR_W - 1)) * 255) * 4;
      ctx.fillStyle = `rgb(${b.table[k]}, ${b.table[k + 1]}, ${b.table[k + 2]})`;
      ctx.fillRect(x + px, barsTop + 18, 1, 12);
    }
    ctx.strokeStyle = t.border;
    ctx.strokeRect(x + 0.5, barsTop + 18.5, BAR_W - 1, 11);
    const ticks: [number, CanvasTextAlign, number][] = [[b.lo, "left", x], [(b.lo + b.hi) / 2, "center", x + BAR_W / 2], [b.hi, "right", x + BAR_W]];
    for (const [v, align, tx] of ticks) text(b.format(v), tx, barsTop + 46, 11.5, t.muted, 400, align);
  });

  text(spec.footer, M, footerY, 11, t.muted);
  return canvas;
}

export function canvasToPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encoding failed"))), "image/png"),
  );
}
