import { useEffect, useRef } from "react";
import { paintFrame } from "../colormap";

export interface Hover {
  ix: number;
  iy: number;
}

interface Props {
  title: string;
  subtitle?: string;
  nx: number;
  ny: number;
  /** Frame values (x-major) or an accessor. */
  values: Float32Array | ((ix: number, iy: number) => number);
  lo: number;
  hi: number;
  table: Uint8ClampedArray;
  hover: Hover | null;
  onHover: (h: Hover | null) => void;
  badge?: string;
  /** Changes whenever `values` must be repainted (accessor identity alone may not change). */
  version: string;
}

/** One 2D field as a pixel-exact heat map, with a crosshair shared across panels. */
export function FieldPanel({
  title, subtitle, nx, ny, values, lo, hi, table, hover, onHover, badge, version,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<ImageData | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    if (!imageRef.current || imageRef.current.width !== nx || imageRef.current.height !== ny) {
      imageRef.current = ctx.createImageData(nx, ny);
    }
    paintFrame(imageRef.current.data, values, nx, ny, lo, hi, table);
    ctx.putImageData(imageRef.current, 0, 0);
  }, [version, nx, ny, lo, hi, table]);

  const toCell = (e: React.PointerEvent<HTMLDivElement>): Hover => {
    const r = e.currentTarget.getBoundingClientRect();
    const ix = Math.floor(((e.clientX - r.left) / r.width) * nx);
    const iy = ny - 1 - Math.floor(((e.clientY - r.top) / r.height) * ny);
    return { ix: Math.min(nx - 1, Math.max(0, ix)), iy: Math.min(ny - 1, Math.max(0, iy)) };
  };

  return (
    <figure className="field-panel">
      <figcaption>
        <span className="field-title">{title}</span>
        {badge && <span className="badge">{badge}</span>}
        {subtitle && <span className="field-sub">{subtitle}</span>}
      </figcaption>
      <div
        className="field-canvas-wrap"
        onPointerMove={(e) => onHover(toCell(e))}
        onPointerDown={(e) => onHover(toCell(e))}
        onPointerLeave={() => onHover(null)}
      >
        <canvas ref={canvasRef} width={nx} height={ny} aria-label={`${title} heat map`} role="img" />
        {hover && (
          <svg className="crosshair" viewBox={`0 0 ${nx} ${ny}`} preserveAspectRatio="none" aria-hidden>
            <line x1={hover.ix + 0.5} x2={hover.ix + 0.5} y1={0} y2={ny} />
            <line x1={0} x2={nx} y1={ny - 1 - hover.iy + 0.5} y2={ny - 1 - hover.iy + 0.5} />
          </svg>
        )}
      </div>
    </figure>
  );
}
