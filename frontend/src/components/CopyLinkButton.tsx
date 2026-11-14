import { useEffect, useState } from "react";

interface Props {
  /** Bring the URL up to date and return it. */
  getUrl: () => string;
  /** Extra note for the confirmation, e.g. what the link cannot carry. */
  note?: string;
}

/** Copies a link to the current view; the result is announced in a status message. */
export function CopyLinkButton({ getUrl, note }: Props) {
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    if (!status) return;
    const id = setTimeout(() => setStatus(null), status.ok ? 2500 : 6000);
    return () => clearTimeout(id);
  }, [status]);

  const copy = async () => {
    const url = getUrl();
    try {
      await navigator.clipboard.writeText(url);
      setStatus({ ok: true, text: note ? `Link copied. ${note}` : "Link copied" });
    } catch {
      setStatus({ ok: false, text: "Could not copy. The address bar has the same link." });
    }
  };

  return (
    <span className="tool">
      <button type="button" className="tool-btn" onClick={copy} title="Copy a link to this sample, time and view">
        <svg viewBox="0 0 16 16" aria-hidden>
          <path d="M6.5 9.5a2.5 2.5 0 0 0 3.5 0l2.5-2.5A2.5 2.5 0 0 0 9 3.5L8 4.5" />
          <path d="M9.5 6.5a2.5 2.5 0 0 0-3.5 0L3.5 9A2.5 2.5 0 0 0 7 12.5l1-1" />
        </svg>
        Copy link
      </button>
      <span className={`tool-status ${status?.ok === false ? "bad" : ""}`} role="status">
        {status?.text}
      </span>
    </span>
  );
}
