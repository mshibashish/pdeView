import { useMemo, useState } from "react";
import { fmtSci } from "../format";
import type { SampleSummary } from "../types";

type SplitFilter = "all" | "test" | "train";
type SortKey = "key" | "radius" | "error";

interface Props {
  samples: SampleSummary[];
  selected: string | null;
  onSelect: (key: string) => void;
}

export function SampleList({ samples, selected, onSelect }: Props) {
  const [split, setSplit] = useState<SplitFilter>("test");
  const [sort, setSort] = useState<SortKey>("key");

  const rows = useMemo(() => {
    const filtered = samples.filter((s) => split === "all" || s.split === split);
    const by: Record<SortKey, (a: SampleSummary, b: SampleSummary) => number> = {
      key: (a, b) => a.key.localeCompare(b.key),
      radius: (a, b) => a.dam_radius - b.dam_radius,
      error: (a, b) => (b.nRMSE ?? 0) - (a.nRMSE ?? 0),
    };
    return [...filtered].sort(by[sort]);
  }, [samples, split, sort]);

  const maxErr = Math.max(...samples.map((s) => s.nRMSE ?? 0), 1e-9);

  return (
    <aside className="sample-list" aria-label="Samples">
      <div className="sample-filters">
        <div className="segmented" role="radiogroup" aria-label="Split">
          {(["test", "train", "all"] as SplitFilter[]).map((s) => (
            <button key={s} type="button" role="radio" aria-checked={split === s}
              className={split === s ? "on" : ""} onClick={() => setSplit(s)}>
              {s === "test" ? "Test" : s === "train" ? "Train probe" : "All"}
            </button>
          ))}
        </div>
        <label className="sort">
          <span>Sort</span>
          <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
            <option value="key">Sample id</option>
            <option value="radius">Dam radius</option>
            <option value="error">FNO error (worst first)</option>
          </select>
        </label>
      </div>
      <div className="sample-head" aria-hidden>
        <span>Sample</span>
        <span>r</span>
        <span>FNO nRMSE</span>
      </div>
      <ul className="sample-rows">
        {rows.map((s) => (
          <li key={s.key}>
            <button
              type="button"
              className={`sample-row ${s.key === selected ? "selected" : ""}`}
              onClick={() => onSelect(s.key)}
              aria-current={s.key === selected}
              title={s.n_twins_total > 0 ? `Bit-identical to ${s.n_twins_total} other sample(s) in the full dataset` : undefined}
            >
              <span className="sample-key">
                {s.key}
                {s.split === "train" && <span className="tag">train</span>}
              </span>
              <span className="sample-r">{s.dam_radius.toFixed(3)}</span>
              <span className="sample-err">
                <span className="err-track" aria-hidden>
                  <span className="err-bar" style={{ width: `${((s.nRMSE ?? 0) / maxErr) * 100}%` }} />
                </span>
                <span className="err-val">{fmtSci(s.nRMSE, 1)}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
      <p className="sample-foot">
        {rows.length} samples · test = PDEBench test keys 0900–0999; train probe = keys 0000–0009.
      </p>
    </aside>
  );
}
