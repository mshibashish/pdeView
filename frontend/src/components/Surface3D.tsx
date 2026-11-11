import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Theme } from "../colormap";

interface Props {
  values: Float32Array; // one [X, Y] frame, x-major
  nx: number;
  ny: number;
  lo: number;
  hi: number;
  table: Uint8ClampedArray;
  theme: Theme;
  version: string;
  exaggeration?: number;
}

const EXTENT = 5; // domain [-2.5, 2.5]^2

/** three.js expects vertex colours in linear space; the LUT holds sRGB bytes. */
const linCache = new WeakMap<Uint8ClampedArray, Float32Array>();
function linearTable(table: Uint8ClampedArray): Float32Array {
  let lin = linCache.get(table);
  if (!lin) {
    lin = new Float32Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      for (let c = 0; c < 3; c++) {
        const s = table[i * 4 + c] / 255;
        lin[i * 3 + c] = s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
      }
    }
    linCache.set(table, lin);
  }
  return lin;
}

/** Water surface h(x, y) as a lit, colour-mapped height field (drag to orbit, scroll to zoom). */
export function Surface3D({ values, nx, ny, lo, hi, table, theme, version, exaggeration = 1.5 }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    geom: THREE.PlaneGeometry;
    raf: number;
  } | null>(null);

  // one-time scene setup
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      mount.textContent = "WebGL is not available in this browser.";
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.position.set(3.7, -4.5, 3.3);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0.1);

    const geom = new THREE.PlaneGeometry(EXTENT, EXTENT, nx - 1, ny - 1);
    geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(nx * ny * 3), 3));
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.55, metalness: 0.0, side: THREE.DoubleSide,
    });
    scene.add(new THREE.Mesh(geom, mat));
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(3, -4, 6);
    scene.add(sun);

    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(EXTENT, EXTENT, 0.001)),
      new THREE.LineBasicMaterial({ color: 0x898781 }),
    );
    scene.add(box);

    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);
    resize();

    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      if (stateRef.current) stateRef.current.raf = requestAnimationFrame(loop);
    };
    stateRef.current = { renderer, scene, camera, controls, geom, raf: 0 };
    stateRef.current.raf = requestAnimationFrame(loop);

    return () => {
      if (stateRef.current) cancelAnimationFrame(stateRef.current.raf);
      stateRef.current = null;
      ro.disconnect();
      controls.dispose();
      geom.dispose();
      mat.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [nx, ny]);

  // update heights + colours whenever the frame changes
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    const pos = st.geom.attributes.position as THREE.BufferAttribute;
    const col = st.geom.attributes.color as THREE.BufferAttribute;
    const scale = 255 / (hi - lo || 1);
    const lin = linearTable(table);
    // PlaneGeometry vertices run row by row from +y (top) to -y, x left to right.
    for (let row = 0; row < ny; row++) {
      const iy = ny - 1 - row;
      for (let ix = 0; ix < nx; ix++) {
        const v = values[ix * ny + iy];
        const idx = row * nx + ix;
        pos.setZ(idx, (v - 1) * exaggeration);
        let k = Math.round((v - lo) * scale);
        k = k < 0 ? 0 : k > 255 ? 255 : k;
        col.setXYZ(idx, lin[k * 3], lin[k * 3 + 1], lin[k * 3 + 2]);
      }
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    st.geom.computeVertexNormals();
  }, [version, lo, hi, table, exaggeration, theme]);

  return <div className="surface3d" ref={mountRef} aria-label="3D water surface (drag to rotate, scroll to zoom)" role="img" />;
}
