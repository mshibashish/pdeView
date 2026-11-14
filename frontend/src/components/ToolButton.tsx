import { useEffect, useState, type ReactNode } from "react";

interface Props {
  icon: ReactNode;
  label: string;
  title?: string;
  /** Does the work and returns the confirmation to show; a thrown error is shown instead. */
  run: () => Promise<string>;
}

/** A view-bar action whose outcome is shown briefly under it and announced as a status message. */
export function ToolButton({ icon, label, title, run }: Props) {
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!status) return;
    const id = setTimeout(() => setStatus(null), status.ok ? 3000 : 6000);
    return () => clearTimeout(id);
  }, [status]);

  const onClick = async () => {
    setBusy(true);
    try {
      setStatus({ ok: true, text: await run() });
    } catch (e) {
      setStatus({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <span className="tool">
      <button type="button" className="tool-btn" onClick={onClick} disabled={busy} title={title}>
        {icon}
        {label}
      </button>
      <span className={`tool-status ${status?.ok === false ? "bad" : ""}`} role="status">
        {status?.text}
      </span>
    </span>
  );
}

export const LinkIcon = () => (
  <svg viewBox="0 0 16 16" aria-hidden>
    <path d="M6.5 9.5a2.5 2.5 0 0 0 3.5 0l2.5-2.5A2.5 2.5 0 0 0 9 3.5L8 4.5" />
    <path d="M9.5 6.5a2.5 2.5 0 0 0-3.5 0L3.5 9A2.5 2.5 0 0 0 7 12.5l1-1" />
  </svg>
);

export const DownloadIcon = () => (
  <svg viewBox="0 0 16 16" aria-hidden>
    <path d="M8 2.5v8M4.5 7 8 10.5 11.5 7M3 13.5h10" />
  </svg>
);
