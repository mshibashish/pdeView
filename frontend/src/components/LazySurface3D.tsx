import { Component, lazy, Suspense, type ReactNode } from "react";
import type { Surface3DProps } from "./Surface3D";

// three.js is most of the bundle, and only the 3D view needs it: load it on first use.
const load = () => import("./Surface3D");
const Impl = lazy(() => load().then((m) => ({ default: m.Surface3D })));

/** Start fetching the 3D view's code early, e.g. when the pointer reaches its button. */
export function preloadSurface3D(): void {
  load().catch(() => {
    /* reported by the boundary below if the view is opened */
  });
}

/** Keeps a failed chunk download from taking the whole viewer down with it. */
class LoadBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };

  static getDerivedStateFromError(e: unknown) {
    return { error: e instanceof Error ? e.message : String(e) };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="surface3d surface-loading" role="alert">
          Could not load the 3D view ({this.state.error}). Reload the page to try again.
        </div>
      );
    }
    return this.props.children;
  }
}

/** The 3D surface view, code-split into its own chunk. */
export function Surface3D(props: Surface3DProps) {
  return (
    <LoadBoundary>
      <Suspense fallback={<div className="surface3d surface-loading" role="status">Loading the 3D view…</div>}>
        <Impl {...props} />
      </Suspense>
    </LoadBoundary>
  );
}
