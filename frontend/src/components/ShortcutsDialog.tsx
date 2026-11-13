import { Fragment, useEffect, useId, useRef } from "react";

/** What the help lists. Keep in step with the useHotkeys bindings in App and SampleList. */
const GROUPS: { title: string; items: [keys: string[][], label: string][] }[] = [
  {
    title: "Playback",
    items: [
      [[["Space"]], "Play or pause"],
      [[["←"], ["→"]], "Previous / next frame"],
      [[["Shift", "←"], ["→"]], "Back / forward 10 frames"],
      [[["Home"], ["End"]], "First / last frame"],
      [[[","], ["."]], "Slower / faster"],
    ],
  },
  {
    title: "View",
    items: [
      [[["2"], ["3"]], "2D maps / 3D surface"],
      [[["C"]], "Colour scale: whole rollout or this frame"],
      [[["J"], ["K"]], "Next / previous sample in the list"],
      [[["↑"], ["↓"]], "Move through the sample list (Enter opens)"],
    ],
  },
  {
    title: "Focused map or chart",
    items: [
      [[["←"], ["↑"], ["→"], ["↓"]], "Move the probe or cursor (Shift: faster)"],
      [[["Enter"]], "Jump to the cursor's time (error and volume charts)"],
      [[["Esc"]], "Release the probe"],
    ],
  },
  {
    title: "General",
    items: [
      [[["?"]], "Show these shortcuts"],
      [[["Esc"]], "Close this dialog"],
    ],
  },
];

function Keys({ combos }: { combos: string[][] }) {
  return (
    <>
      {combos.map((combo, i) => (
        <Fragment key={combo.join("+")}>
          {i > 0 && <span className="kbd-sep">/</span>}
          {combo.map((k, j) => (
            <Fragment key={k}>
              {j > 0 && <span className="kbd-sep">+</span>}
              <kbd>{k}</kbd>
            </Fragment>
          ))}
        </Fragment>
      ))}
    </>
  );
}

/** Modal list of keyboard shortcuts (native <dialog>: focus is trapped and restored, Esc closes). */
export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    else if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="shortcuts"
      aria-labelledby={titleId}
      onClose={onClose}
      // a click on the backdrop lands on the <dialog> itself; the content sits in the inner div
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="shortcuts-body">
        <div className="shortcuts-head">
          <h2 id={titleId}>Keyboard shortcuts</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            <svg viewBox="0 0 16 16" aria-hidden><path d="M4 4l8 8M12 4l-8 8" /></svg>
          </button>
        </div>
        <div className="shortcuts-groups">
          {GROUPS.map((g) => (
            <section key={g.title}>
              <h3>{g.title}</h3>
              <dl>
                {g.items.map(([combos, label]) => (
                  <div key={label}>
                    <dt><Keys combos={combos} /></dt>
                    <dd>{label}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </div>
    </dialog>
  );
}
