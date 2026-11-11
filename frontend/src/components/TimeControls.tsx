interface Props {
  frame: number;
  nFrames: number;
  t: number[];
  initialStep: number;
  playing: boolean;
  speed: number;
  onFrame: (f: number) => void;
  onToggle: () => void;
  onStep: (d: number) => void;
  onSpeed: (s: number) => void;
}

const SPEEDS = [0.25, 0.5, 1, 2, 4];

export function TimeControls({
  frame, nFrames, t, initialStep, playing, speed, onFrame, onToggle, onStep, onSpeed,
}: Props) {
  const inputPct = ((initialStep - 0.5) / (nFrames - 1)) * 100;
  const isInput = frame < initialStep;
  return (
    <div className="time-controls">
      <div className="time-buttons">
        <button type="button" className="icon-btn" onClick={() => onStep(-1)} aria-label="Previous frame" title="Previous frame (←)">
          <svg viewBox="0 0 16 16" aria-hidden><path d="M10 3 5 8l5 5" /></svg>
        </button>
        <button type="button" className="icon-btn primary" onClick={onToggle} aria-label={playing ? "Pause" : "Play"} title="Play / pause (space)">
          {playing ? (
            <svg viewBox="0 0 16 16" aria-hidden><path d="M5 3v10M11 3v10" /></svg>
          ) : (
            <svg viewBox="0 0 16 16" aria-hidden><path d="M5 3l8 5-8 5z" className="fill" /></svg>
          )}
        </button>
        <button type="button" className="icon-btn" onClick={() => onStep(1)} aria-label="Next frame" title="Next frame (→)">
          <svg viewBox="0 0 16 16" aria-hidden><path d="m6 3 5 5-5 5" /></svg>
        </button>
      </div>
      <div className="time-track">
        <div className="time-regions" aria-hidden>
          <div className="region input" style={{ width: `${inputPct}%` }}>
            <span>GT input</span>
          </div>
          <div className="region rollout">
            <span>FNO autoregressive rollout →</span>
          </div>
        </div>
        <input
          type="range"
          min={0}
          max={nFrames - 1}
          step={1}
          value={frame}
          onChange={(e) => onFrame(Number(e.target.value))}
          aria-label="Time frame"
          aria-valuetext={`t = ${t[frame]?.toFixed(2)}, frame ${frame}`}
        />
      </div>
      <div className="time-readout">
        <span className="time-value">t = {t[frame]?.toFixed(2)}</span>
        <span className="time-frame">
          frame {frame}/{nFrames - 1} · {isInput ? "ground-truth input" : `prediction step ${frame - initialStep + 1}`}
        </span>
      </div>
      <label className="speed">
        <span>Speed</span>
        <select value={speed} onChange={(e) => onSpeed(Number(e.target.value))}>
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
