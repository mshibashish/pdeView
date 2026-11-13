import { useEffect, useMemo, useRef, useState } from "react";
import { fmtSci } from "../format";
import { useHotkeys } from "../hooks/useHotkeys";
import type { SampleSummary } from "../types";

type SplitFilter = "all" | "test" | "heldout" | "train";
type SortKey = "key" | "radius" | "error";

interface Props {
  samples: SampleSummary[];
  selected: string | null;
  onSelect: (key: string) => void;
}

const FILTERS: [SplitFilter, string][] = [
  ["test", "Test"], ["heldout", "Held-out"], ["train", "Train probe"], ["all", "All"],
];

export function SampleList({ samples, selected, onSelect }: Props) {
  const [split, setSplit] = useState<SplitFilter>("test");
  const hasHeldout = samples.some((s) => s.split === "heldout");
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

  const maxErr = Math.max(...rows.map((s) => s.nRMSE ?? 0), 1e-9); // bars scale to the visible rows
  const listRef = useRef<HTMLUListElement>(null);
  const selectedIdx = rows.findIndex((s) => s.key === selected);
  const tabStop = Math.max(0, selectedIdx); // one tab stop for the whole list; arrows move within it

  // J / K: next / previous sample in the visible order
  const go = (d: number) => {
    const next = rows[selectedIdx < 0 ? 0 : Math.min(rows.length - 1, Math.max(0, selectedIdx + d))];
    if (next) onSelect(next.key);
  };
  useHotkeys({ j: () => go(1), k: () => go(-1) });

  // Keep the selected row in view by scrolling the list only (never the page).
  useEffect(() => {
    const ul = listRef.current;
    const row = ul?.children[selectedIdx] as HTMLElement | undefined;
    if (!ul || !row) return;
    if (row.offsetTop < ul.scrollTop) ul.scrollTop = row.offsetTop;
    else if (row.offsetTop + row.offsetHeight > ul.scrollTop + ul.clientHeight) {
      ul.scrollTop = row.offsetTop + row.offsetHeight - ul.clientHeight;
    }
  }, [selectedIdx]);

  // Up / Down / Home / End move focus between rows; Enter or Space opens the focused sample.
  const onListKey = (e: React.KeyboardEvent<HTMLUListElement>) => {
    const buttons = [...e.currentTarget.querySelectorAll<HTMLButtonElement>("button.sample-row")];
    const i = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const to = { ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: buttons.length - 1 }[e.key];
    if (i < 0 || to === undefined || e.altKey || e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    buttons[Math.min(buttons.length - 1, Math.max(0, to))]?.focus();
  };

  return (
    <aside className="sample-list" aria-label="Samples">
      <div className="sample-filters">
        <div className="segmented" role="radiogroup" aria-label="Split">
          {FILTERS.filter(([s]) => s !== "heldout" || hasHeldout).map(([s, label]) => (
            <button key={s} type="button" role="radio" aria-checked={split === s}
              className={split === s ? "on" : ""} onClick={() => setSplit(s)}>
              {label}
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
      <ul className="sample-rows" ref={listRef} onKeyDown={onListKey}>
        {rows.map((s, i) => (
          <li key={s.key}>
            <button
              type="button"
              tabIndex={i === tabStop ? 0 : -1}
              className={`sample-row ${s.key === selected ? "selected" : ""}`}
              onClick={() => onSelect(s.key)}
              aria-current={s.key === selected}
              title={s.split === "heldout"
                ? `Held-out case (${s.family === "subcell" ? "dam centre shifted within half a cell" : "cell-averaged dam wall"}); its initial condition never occurs in PDEBench's training split`
                : s.n_twins_total > 0 ? `Bit-identical to ${s.n_twins_total} other sample(s) in the full dataset` : undefined}
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
        {rows.length} samples · test = PDEBench test keys 0900–0999; train probe = keys 0000–0009
        {hasHeldout && "; held-out = new PyClaw runs whose initial conditions never occur in training"}.
      </p>
    </aside>
  );
}
