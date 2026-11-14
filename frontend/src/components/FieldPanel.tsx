import { useEffect, useId, useRef } from "react";
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
  onHover: React.Dispatch<React.SetStateAction<Hover | null>>;
  badge?: string;
  /** Changes whenever `values` must be repainted (accessor identity alone may not change). */
  version: string;
}

const KEY_STEP: Record<string, [number, number]> = {
  ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1],
};

/**
 * One 2D field as a pixel-exact heat map, with a crosshair shared across panels. The probe follows the
 * pointer, or the arrow keys while the map has focus (Shift moves 8 cells).
 */
export function FieldPanel({
  title, subtitle, nx, ny, values, lo, hi, table, hover, onHover, badge, version,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const helpId = useId();
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

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.currentTarget.blur();
      return;
    }
    const d = KEY_STEP[e.key];
    if (!d || e.altKey || e.ctrlKey || e.metaKey) return;
    e.preventDefault(); // also tells the global shortcuts that this key is taken
    const n = e.shiftKey ? 8 : 1;
    onHover((h) => {
      const c = h ?? { ix: nx >> 1, iy: ny >> 1 };
      return {
        ix: Math.min(nx - 1, Math.max(0, c.ix + d[0] * n)),
        iy: Math.min(ny - 1, Math.max(0, c.iy + d[1] * n)),
      };
    });
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
        tabIndex={0}
        role="application"
        aria-roledescription="heat map"
        aria-label={title}
        aria-describedby={helpId}
        onPointerMove={(e) => onHover(toCell(e))}
        onPointerDown={(e) => onHover(toCell(e))}
        onPointerLeave={(e) => {
          if (document.activeElement !== e.currentTarget) onHover(null);
        }}
        onFocus={() => onHover((h) => h ?? { ix: nx >> 1, iy: ny >> 1 })}
        onBlur={() => onHover(null)}
        onKeyDown={onKeyDown}
      >
        <span id={helpId} className="visually-hidden">
          Arrow keys move the probe, Shift moves 8 cells. Its values are read out below the maps.
        </span>
        <canvas ref={canvasRef} width={nx} height={ny} aria-hidden />
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
