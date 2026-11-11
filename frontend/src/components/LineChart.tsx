import { useEffect, useMemo, useRef, useState } from "react";

export interface Series {
  id: string;
  label: string;
  color: string; // CSS colour or var(--token)
  values: (number | null)[];
  dash?: string;
}

interface Props {
  title: string;
  x: number[];
  series: Series[];
  yScale?: "linear" | "log";
  xLabel: string;
  yLabel?: string;
  formatY: (v: number) => string;
  formatX?: (v: number) => string;
  marker?: number; // x value of the "current time" marker
  markerLabel?: string;
  height?: number;
  yDomain?: [number, number];
  onPick?: (index: number) => void; // click to seek
  shade?: [number, number]; // x-range to shade (e.g. the GT input frames)
  shadeLabel?: string;
  formatTip?: (v: number) => string; // tooltip value format (defaults to formatY)
  formatTipX?: (v: number) => string; // tooltip x format (defaults to formatX)
  leftMargin?: number;
}

const MARGIN = { top: 8, right: 14, bottom: 34, left: 54 };

function niceTicks(lo: number, hi: number, n = 4): number[] {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.5) ?? mag * 10;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return ticks;
}

function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    const v = 10 ** e;
    if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
  }
  return out;
}

/** Small responsive SVG line chart: hairline grid, 2px lines, crosshair + tooltip. */
export function LineChart({
  title, x, series, yScale = "linear", xLabel, yLabel, formatY, formatX = (v) => v.toFixed(1),
  marker, markerLabel, height = 200, yDomain, onPick, shade, shadeLabel, formatTip, formatTipX,
  leftMargin,
}: Props) {
  const M = { ...MARGIN, left: leftMargin ?? MARGIN.left };
  const tipY = formatTip ?? formatY;
  const tipX = formatTipX ?? formatX;
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(360);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(220, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;

  const { y0, y1 } = useMemo(() => {
    if (yDomain) return { y0: yDomain[0], y1: yDomain[1] };
    const vals = series.flatMap((s) => s.values).filter((v): v is number =>
      v !== null && Number.isFinite(v) && (yScale === "linear" || v > 0));
    if (!vals.length) return { y0: 0, y1: 1 };
    let lo = Math.min(...vals);
    let hi = Math.max(...vals);
    if (yScale === "log") {
      return { y0: 10 ** Math.floor(Math.log10(lo)), y1: 10 ** Math.ceil(Math.log10(hi)) };
    }
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1;
    lo -= pad;
    hi += pad;
    return { y0: lo, y1: hi };
  }, [series, yScale, yDomain]);

  const xs = (v: number) => M.left + ((v - x[0]) / (x[x.length - 1] - x[0] || 1)) * iw;
  const ys = (v: number) =>
    yScale === "log"
      ? M.top + (1 - (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * ih
      : M.top + (1 - (v - y0) / (y1 - y0 || 1)) * ih;

  const yTicks = yScale === "log" ? logTicks(y0, y1) : niceTicks(y0, y1);
  const xTicks = niceTicks(x[0], x[x.length - 1], 5);

  const paths = series.map((s) => {
    let d = "";
    let pen = false;
    s.values.forEach((v, i) => {
      if (v === null || !Number.isFinite(v) || (yScale === "log" && v <= 0)) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${xs(x[i]).toFixed(1)},${ys(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  });

  const indexAt = (clientX: number, rect: DOMRect) => {
    const px = clientX - rect.left - M.left;
    const v = x[0] + (px / iw) * (x[x.length - 1] - x[0]);
    let best = 0;
    for (let i = 1; i < x.length; i++) if (Math.abs(x[i] - v) < Math.abs(x[best] - v)) best = i;
    return best;
  };

  const hx = hoverIdx !== null ? xs(x[hoverIdx]) : null;
  const tooltipLeft = hx !== null && hx + 200 > width;

  return (
    <div className="chart" ref={wrapRef}>
      <div className="chart-head">
        <div className="chart-title">{title}</div>
        <ul className="legend" aria-label="Legend">
          {series.map((s) => (
            <li key={s.id}>
              <svg width="18" height="8" aria-hidden>
                <line x1="1" x2="17" y1="4" y2="4" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} strokeLinecap="round" />
              </svg>
              {s.label}
            </li>
          ))}
        </ul>
      </div>
      <div className="chart-body">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={title}
          onPointerMove={(e) => setHoverIdx(indexAt(e.clientX, e.currentTarget.getBoundingClientRect()))}
          onPointerLeave={() => setHoverIdx(null)}
          onClick={(e) => onPick?.(indexAt(e.clientX, e.currentTarget.getBoundingClientRect()))}
          style={{ cursor: onPick ? "pointer" : undefined }}
        >
          {shade && (
            <g>
              <rect className="chart-shade" x={xs(shade[0])} y={M.top} width={Math.max(0, xs(shade[1]) - xs(shade[0]))} height={ih} />
              {shadeLabel && (
                <text className="chart-shade-label" x={xs(shade[0]) + 4} y={M.top + 12}>{shadeLabel}</text>
              )}
            </g>
          )}
          {yTicks.map((v) => (
            <g key={`y${v}`}>
              <line className="grid" x1={M.left} x2={M.left + iw} y1={ys(v)} y2={ys(v)} />
              <text className="tick" x={M.left - 6} y={ys(v)} dy="0.32em" textAnchor="end">{formatY(v)}</text>
            </g>
          ))}
          {xTicks.map((v) => (
            <text key={`x${v}`} className="tick" x={xs(v)} y={M.top + ih + 16} textAnchor="middle">{formatX(v)}</text>
          ))}
          <line className="axis" x1={M.left} x2={M.left + iw} y1={M.top + ih} y2={M.top + ih} />
          <text className="axis-label" x={M.left + iw / 2} y={height - 4} textAnchor="middle">{xLabel}</text>
          {yLabel && (
            <text className="axis-label" transform={`translate(12 ${M.top + ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
          )}
          {marker !== undefined && (
            <g>
              <line className="marker" x1={xs(marker)} x2={xs(marker)} y1={M.top} y2={M.top + ih} />
              {markerLabel && (
                <text className="marker-label" x={xs(marker) + 4} y={M.top + ih - 6}>{markerLabel}</text>
              )}
            </g>
          )}
          {paths.map((d, i) => (
            <path key={series[i].id} d={d} fill="none" stroke={series[i].color} strokeWidth={2}
              strokeLinejoin="round" strokeLinecap="round" strokeDasharray={series[i].dash} />
          ))}
          {hx !== null && hoverIdx !== null && (
            <g>
              <line className="crosshair-line" x1={hx} x2={hx} y1={M.top} y2={M.top + ih} />
              {series.map((s) => {
                const v = s.values[hoverIdx];
                if (v === null || !Number.isFinite(v) || (yScale === "log" && v <= 0)) return null;
                return <circle key={s.id} cx={hx} cy={ys(v)} r={4} fill={s.color} className="dot" />;
              })}
            </g>
          )}
        </svg>
        {hoverIdx !== null && hx !== null && (
          <div
            className="tooltip"
            style={tooltipLeft ? { right: width - hx + 10, top: M.top } : { left: hx + 10, top: M.top }}
          >
            <div className="tooltip-x">{xLabel.split(" ")[0]} = {tipX(x[hoverIdx])}</div>
            {series.map((s) => {
              const v = s.values[hoverIdx];
              return (
                <div className="tooltip-row" key={s.id}>
                  <svg width="14" height="8" aria-hidden>
                    <line x1="1" x2="13" y1="4" y2="4" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} />
                  </svg>
                  <strong>{v === null || !Number.isFinite(v) ? "–" : tipY(v)}</strong>
                  <span>{s.label}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
