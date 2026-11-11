import { useCallback, useEffect, useRef, useState } from "react";

/** Frame index state with requestAnimationFrame-driven playback. */
export function usePlayback(nFrames: number, baseFps = 20) {
  const [frame, setFrameState] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const frameRef = useRef(0);

  const setFrame = useCallback(
    (f: number) => {
      const clamped = Math.min(Math.max(0, nFrames - 1), Math.max(0, Math.round(f)));
      frameRef.current = clamped;
      setFrameState(clamped);
    },
    [nFrames],
  );

  useEffect(() => {
    if (!playing || nFrames < 2) return;
    let raf = 0;
    let last: number | null = null;
    let acc = 0;
    const tick = (now: number) => {
      if (last !== null) {
        acc += ((now - last) / 1000) * baseFps * speed;
        const steps = Math.floor(acc);
        if (steps > 0) {
          acc -= steps;
          const next = Math.min(nFrames - 1, frameRef.current + steps);
          setFrame(next);
          if (next >= nFrames - 1) {
            setPlaying(false);
            return;
          }
        }
      }
      last = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, nFrames, baseFps, setFrame]);

  const toggle = useCallback(() => {
    if (!playing && frameRef.current >= nFrames - 1) setFrame(0);
    setPlaying(!playing);
  }, [playing, nFrames, setFrame]);

  const step = useCallback(
    (d: number) => {
      setPlaying(false);
      setFrame(frameRef.current + d);
    },
    [setFrame],
  );

  return { frame, setFrame, playing, setPlaying, toggle, step, speed, setSpeed };
}
