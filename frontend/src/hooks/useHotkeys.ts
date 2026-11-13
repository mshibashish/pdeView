import { useEffect, useRef } from "react";

export type Hotkeys = Record<string, (e: KeyboardEvent) => void>;

const NATIVE_SPACE = new Set(["BUTTON", "A", "SUMMARY"]);
const RANGE_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);

/**
 * Whether a global shortcut should leave this key press alone: it carries a browser modifier, a
 * focused control needs it (typing, a slider's arrows, Space on a button), it comes from inside a
 * modal dialog, or a component has already handled it (preventDefault).
 */
export function ignoreKey(e: KeyboardEvent): boolean {
  if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return true;
  const el = e.target instanceof HTMLElement ? e.target : null;
  if (!el) return false;
  if (el.closest("dialog")) return true;
  if (el.isContentEditable || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return true;
  if (el instanceof HTMLInputElement) return el.type !== "range" || RANGE_KEYS.has(e.key);
  return e.key === " " && NATIVE_SPACE.has(el.tagName);
}

/**
 * Window-level shortcuts keyed by `KeyboardEvent.key` (letters lower-case). A matched key is marked
 * handled. The latest bindings are always used, so handlers can read fresh state.
 */
export function useHotkeys(bindings: Hotkeys, enabled = true): void {
  const ref = useRef(bindings);
  useEffect(() => {
    ref.current = bindings;
  });
  useEffect(() => {
    if (!enabled) return;
    const on = (e: KeyboardEvent) => {
      if (ignoreKey(e)) return;
      const fn = ref.current[e.key.length === 1 ? e.key.toLowerCase() : e.key];
      if (!fn) return;
      e.preventDefault();
      fn(e);
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [enabled]);
}
