import { gradientCss, type Theme } from "../colormap";

interface Props {
  kind: "height" | "error";
  theme: Theme;
  lo: number;
  hi: number;
  label: string;
  format?: (v: number) => string;
}

export function Colorbar({ kind, theme, lo, hi, label, format = (v) => v.toFixed(2) }: Props) {
  const mid = (lo + hi) / 2;
  return (
    <div className="colorbar">
      <div className="colorbar-label">{label}</div>
      <div className="colorbar-bar" style={{ background: gradientCss(kind, theme) }} />
      <div className="colorbar-ticks">
        <span>{format(lo)}</span>
        <span>{format(mid)}</span>
        <span>{format(hi)}</span>
      </div>
    </div>
  );
}
