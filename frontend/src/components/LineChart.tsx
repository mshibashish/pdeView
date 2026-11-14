import { useEffect, useId, useMemo, useRef, useState } from "react";
import { csvBlob, downloadBlob } from "../download";

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
  csvName?: string; // offer the plotted data as a CSV download under this file name
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
  leftMargin, csvName,
}: Props) {
  const M = { ...MARGIN, left: leftMargin ?? MARGIN.left };
  const tipY = formatTip ?? formatY;
  const tipX = formatTipX ?? formatX;
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(360);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [fromKeys, setFromKeys] = useState(false); // announce keyboard moves only, not pointer hover
  const helpId = useId();

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

  const nearest = (v: number) => {
    let best = 0;
    for (let i = 1; i < x.length; i++) if (Math.abs(x[i] - v) < Math.abs(x[best] - v)) best = i;
    return best;
  };
  const indexAt = (clientX: number, rect: DOMRect) =>
    nearest(x[0] + ((clientX - rect.left - M.left) / iw) * (x[x.length - 1] - x[0]));

  // Keyboard: the same crosshair and tooltip as hover. Arrows move it (Shift: 10 points), Home/End
  // jump to the ends, Enter or Space seeks when the chart supports it.
  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const cur = hoverIdx ?? (marker !== undefined ? nearest(marker) : 0);
    const n = e.shiftKey ? 10 : 1;
    const moves: Record<string, number> = {
      ArrowLeft: cur - n, ArrowRight: cur + n, Home: 0, End: x.length - 1,
    };
    if (e.key in moves) {
      setHoverIdx(Math.min(x.length - 1, Math.max(0, moves[e.key])));
      setFromKeys(true);
    } else if ((e.key === "Enter" || e.key === " ") && onPick) {
      onPick(cur);
    } else if (e.key === "Escape" && hoverIdx !== null) {
      setHoverIdx(null);
    } else {
      return;
    }
    e.preventDefault(); // also tells the global shortcuts that this key is taken
  };
  const describe = (i: number) =>
    `${xLabel.split(" ")[0]} = ${tipX(x[i])}: ${series
      .map((s) => `${s.label} ${s.values[i] === null || !Number.isFinite(s.values[i]) ? "no value" : tipY(s.values[i] as number)}`)
      .join(", ")}`;

  // The plotted numbers, one row per x, at float32 precision (7 significant digits).
  const saveCsv = (name: string) => {
    const num = (v: number | null) => (v === null || !Number.isFinite(v) ? null : Number(v.toPrecision(7)));
    const head = [xLabel, ...series.map((s) => (yLabel ? `${s.label} ${yLabel}` : s.label))];
    downloadBlob(csvBlob(head, x.map((xv, i) => [num(xv), ...series.map((s) => num(s.values[i]))])), name);
  };

  const hx = hoverIdx !== null ? xs(x[hoverIdx]) : null;
  const tooltipLeft = hx !== null && hx + 200 > width;

  return (
    <div className="chart" ref={wrapRef}>
      <div className="chart-head">
        <div className="chart-title">{title}</div>
        {csvName && (
          <button type="button" className="chart-btn" onClick={() => saveCsv(csvName)}
            aria-label={`Download the data of "${title}" as CSV`} title="Download the plotted data (CSV)">
            <svg viewBox="0 0 16 16" aria-hidden><path d="M8 2.5v8M4.5 7 8 10.5 11.5 7M3 13.5h10" /></svg>
            CSV
          </button>
        )}
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
      <div
        className="chart-body"
        tabIndex={0}
        role="application"
        aria-roledescription="chart"
        aria-label={title}
        aria-describedby={helpId}
        onKeyDown={onKeyDown}
        onFocus={() => setHoverIdx((i) => i ?? (marker !== undefined ? nearest(marker) : 0))}
        onBlur={() => {
          setHoverIdx(null);
          setFromKeys(false);
        }}
      >
        <span id={helpId} className="visually-hidden">
          Left and right arrows move the cursor, Shift moves 10 points{onPick ? ", Enter jumps to that time" : ""}.
        </span>
        <span className="visually-hidden" aria-live="polite">
          {fromKeys && hoverIdx !== null ? describe(hoverIdx) : ""}
        </span>
        <svg
          width={width}
          height={height}
          aria-hidden
          onPointerMove={(e) => {
            setHoverIdx(indexAt(e.clientX, e.currentTarget.getBoundingClientRect()));
            setFromKeys(false);
          }}
          onPointerLeave={() => {
            if (!fromKeys) setHoverIdx(null);
          }}
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
