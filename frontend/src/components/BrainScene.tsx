import { useEffect, useRef, useState } from 'react';
import { assetUrl } from '../lib/assets';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export interface BrainProps {
  values: Record<string, number | null>;
  selected: string | null;
  autoRotate: boolean;
  glow: boolean;
  playing: boolean;
  visible: boolean;
  view: 'perspective' | 'front' | 'side' | 'top';
  resetKey: number;
  reducedMotion: boolean;
  onHover: (key: string | null) => void;
  onSelect: (key: string | null) => void;
  onReady: () => void;
  onError: () => void;
  /** Optional clock/speed for the explicitly labeled concept film, not analysis data. */
  illustrationTime?: number;
  rotationSpeed?: number;
  ariaLabel?: string;
}

const vertexShader = `
attribute float region;
attribute float sulc;
uniform float signals[7];
uniform float valid[7];
uniform float selected;
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vView;
varying float vSignal;
varying float vValid;
varying float vSulc;
varying float vFocus;
void main() {
  int r = int(clamp(region, 0.0, 6.0));
  vSignal = region < 0.0 ? 0.0 : signals[r];
  vValid = region < 0.0 ? 0.0 : valid[r];
  vFocus = selected < -0.5 || abs(region - selected) < 0.5 ? 1.0 : 0.10;
  vNormal = normalize(normalMatrix * normal);
  vPosition = position;
  vSulc = sulc;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vView = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}`;

const fragmentShader = `
uniform float glow;
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vView;
varying float vSignal;
varying float vValid;
varying float vSulc;
varying float vFocus;
void main() {
  vec3 n = normalize(vNormal);
  float key = max(dot(n, normalize(vec3(-0.4, 0.8, 1.0))), 0.0);
  float fill = max(dot(n, normalize(vec3(1.0, -0.2, 0.3))), 0.0);
  float rim = pow(1.0 - max(dot(n, normalize(vView)), 0.0), 2.7);
  float fold = 0.55 + 0.45 * (1.0 - smoothstep(-1.0, 1.4, vSulc));
  vec3 base = vec3(0.065, 0.095, 0.135) * (0.27 + key * 1.25 + fill * 0.23) * fold;
  base += vec3(0.24, 0.34, 0.40) * rim * 0.55;
  float strength = smoothstep(0.18, 2.3, abs(vSignal)) * vValid * vFocus;
  vec3 warm = vec3(1.0, 0.30, 0.055);
  vec3 cool = vec3(0.05, 0.39, 0.82);
  vec3 signalColor = vSignal >= 0.0 ? warm : cool;
  vec3 color = mix(base, signalColor * (0.38 + 0.6 * key) * fold, strength * 0.88);
  color += signalColor * pow(strength, 1.5) * (0.35 + glow * 0.95) * fold;
  color += vec3(0.60, 0.72, 0.81) * rim * 0.20;
  gl_FragColor = vec4(color, 1.0);
}`;

const pointVertex = `
attribute float region;
uniform float signals[7];
uniform float valid[7];
uniform float selected;
uniform float phase;
uniform float pixelRatio;
varying float vAlpha;
varying float vSignal;
void main() {
  int r = int(clamp(region, 0.0, 6.0));
  vSignal = region < 0.0 ? 0.0 : signals[r];
  float enabled = region < 0.0 ? 0.0 : valid[r];
  float focus = selected < -0.5 || abs(region - selected) < 0.5 ? 1.0 : 0.03;
  float seed = fract(sin(dot(position.xyz, vec3(12.9898,78.233,45.164))) * 43758.5453);
  float shimmer = 0.6 + 0.4 * sin(phase * 1.6 + seed * 40.0);
  vAlpha = smoothstep(0.4, 2.2, abs(vSignal)) * enabled * focus * shimmer * step(0.66, seed);
  vec4 mv = modelViewMatrix * vec4(position + normal * 0.35, 1.0);
  gl_PointSize = min(6.0, (1.4 + vAlpha * 2.3) * pixelRatio * 240.0 / -mv.z);
  gl_Position = projectionMatrix * mv;
}`;

const pointFragment = `
varying float vAlpha;
varying float vSignal;
void main() {
  float d = length(gl_PointCoord - vec2(0.5)) * 2.0;
  if (d > 1.0 || vAlpha < 0.01) discard;
  vec3 c = vSignal >= 0.0 ? vec3(1.0, 0.62, 0.25) : vec3(0.25, 0.70, 1.0);
  gl_FragColor = vec4(c * 2.0, exp(-d*d*4.0) * vAlpha * 0.85);
}`;

export default function BrainScene(props: BrainProps) {
  const host = useRef<HTMLDivElement>(null);
  const live = useRef(props);
  live.current = props;
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const element = host.current!;
    let disposed = false;
    let animationId = 0;
    let renderer: THREE.WebGLRenderer | undefined;
    let composer: EffectComposer | undefined;
    let controls: OrbitControls | undefined;
    let observer: ResizeObserver | undefined;
    const abort = new AbortController();
    const disposables: { dispose: () => void }[] = [];
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1018);
    const camera = new THREE.PerspectiveCamera(33, 1, 1, 1500);
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let mesh: THREE.Mesh | undefined;
    let regionIds: Float32Array | undefined;
    let keys: string[] = [];
    let down = { x: 0, y: 0 };

    function pick(event: PointerEvent) {
      if (!mesh || !regionIds) return null;
      const bounds = element.getBoundingClientRect();
      mouse.set((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1);
      raycaster.setFromCamera(mouse, camera);
      const hit = raycaster.intersectObject(mesh)[0];
      if (!hit?.face) return null;
      const region = regionIds[hit.face.a];
      return region < 0 ? null : keys[region] ?? null;
    }
    function move(e: PointerEvent) { live.current.onHover(pick(e)); }
    function leave() { live.current.onHover(null); }
    function pointerDown(e: PointerEvent) { down = { x: e.clientX, y: e.clientY }; }
    function pointerUp(e: PointerEvent) {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4) {
        const key = pick(e);
        if (key) live.current.onSelect(live.current.selected === key ? null : key);
      }
    }
    function contextLost(event: Event) { event.preventDefault(); live.current.onError(); }

    async function initialize() {
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
        renderer.setClearColor(0x080c12, 0);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.15;
        renderer.domElement.setAttribute('aria-label', live.current.ariaLabel ?? 'Interactive 3D cortical brain. Drag to rotate; use the view buttons for keyboard navigation.');
        renderer.domElement.setAttribute('role', 'img');
        element.appendChild(renderer.domElement);
        renderer.domElement.addEventListener('webglcontextlost', contextLost);
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.07;
        controls.enablePan = false;
        controls.enableZoom = false;
        controls.autoRotateSpeed = 0.6;
        controls.minPolarAngle = Math.PI * 0.14;
        controls.maxPolarAngle = Math.PI * 0.87;
        camera.position.set(245, 115, 280);
        controls.target.set(0, 0, 0);
        controls.update();

        const [metaResponse, binaryResponse] = await Promise.all([
          fetch(assetUrl('brain/brain.meta.json'), { signal: abort.signal }),
          fetch(assetUrl('brain/brain.bin'), { signal: abort.signal }),
        ]);
        if (!metaResponse.ok || !binaryResponse.ok) throw new Error('Could not load the cortical surface.');
        const [meta, binary] = await Promise.all([metaResponse.json(), binaryResponse.arrayBuffer()]);
        if (disposed) return;
        keys = meta.channel_keys;
        if (keys.length !== 7 || keys.some(key => !(key in live.current.values))) throw new Error('Incompatible demo mapping.');
        const header = new Uint32Array(binary, 0, 2);
        const count = header[0], faceCount = header[1];
        let offset = 8;
        const position = new Float32Array(binary, offset, count * 3); offset += count * 12;
        const sulc = new Float32Array(binary, offset, count); offset += count * 4;
        regionIds = new Float32Array(binary, offset, count); offset += count * 4;
        const indices = new Uint32Array(binary, offset, faceCount * 3);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
        geometry.setAttribute('sulc', new THREE.BufferAttribute(sulc, 1));
        geometry.setAttribute('region', new THREE.BufferAttribute(regionIds, 1));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        disposables.push(geometry);
        const uniforms = {
          signals: { value: new Float32Array(7) }, valid: { value: new Float32Array(7) },
          selected: { value: -1 }, glow: { value: 1 }, phase: { value: 0 },
          pixelRatio: { value: renderer.getPixelRatio() },
        };
        const material = new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms });
        const pointsMaterial = new THREE.ShaderMaterial({
          vertexShader: pointVertex, fragmentShader: pointFragment, uniforms,
          transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
        });
        mesh = new THREE.Mesh(geometry, material);
        const points = new THREE.Points(geometry, pointsMaterial);
        scene.add(mesh, points);
        disposables.push(material, pointsMaterial);
        composer = new EffectComposer(renderer);
        composer.addPass(new RenderPass(scene, camera));
        const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.60, 0.6, 0.85);
        composer.addPass(bloom);
        const output = new OutputPass();
        composer.addPass(output);
        disposables.push(bloom, output);

        function resize() {
          const width = element.clientWidth, height = element.clientHeight;
          if (!width || !height || !renderer || !composer) return;
          camera.aspect = width / height;
          camera.fov = width < 560 ? 40 : 33;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
          composer.setSize(width, height);
        }
        observer = new ResizeObserver(resize);
        observer.observe(element);
        resize();
        let previous = performance.now();
        let view = live.current.view;
        let resetKey = live.current.resetKey;
        const presets = { perspective: [245, 115, 280], front: [0, 20, -390], side: [390, 20, 0], top: [0, 390, 0.1] };
        function frame(now: number) {
          if (disposed || !composer || !controls) return;
          const state = live.current;
          const delta = Math.min((now - previous) / 1000, 0.05);
          previous = now;
          keys.forEach((key, i) => {
            const value = state.values[key];
            uniforms.signals.value[i] = value ?? 0;
            uniforms.valid.value[i] = value == null ? 0 : 1;
          });
          uniforms.selected.value = state.selected ? keys.indexOf(state.selected) : -1;
          uniforms.glow.value = state.glow ? 1 : 0;
          bloom.strength = state.glow ? 0.60 : 0;
          points.visible = state.glow;
          if (state.illustrationTime !== undefined) uniforms.phase.value = state.reducedMotion ? 0 : state.illustrationTime * 2;
          else if (state.playing && !state.reducedMotion) uniforms.phase.value += delta;
          if (view !== state.view || resetKey !== state.resetKey) {
            view = state.view; resetKey = state.resetKey;
            camera.position.fromArray(presets[view]);
            controls.target.set(0, 0, 0);
          }
          if (state.visible && !document.hidden) {
            controls.autoRotate = state.autoRotate && !state.reducedMotion;
            controls.autoRotateSpeed = state.rotationSpeed ?? 0.6;
            controls.update(delta);
            composer.render();
          }
          animationId = requestAnimationFrame(frame);
        }
        element.addEventListener('pointermove', move);
        element.addEventListener('pointerleave', leave);
        element.addEventListener('pointerdown', pointerDown);
        element.addEventListener('pointerup', pointerUp);
        setReady(true);
        live.current.onReady();
        animationId = requestAnimationFrame(frame);
      } catch (error) {
        if (!disposed && !(error instanceof DOMException && error.name === 'AbortError')) live.current.onError();
      }
    }
    void initialize();
    return () => {
      disposed = true;
      abort.abort();
      cancelAnimationFrame(animationId);
      observer?.disconnect();
      controls?.dispose();
      disposables.forEach(d => d.dispose());
      composer?.dispose();
      renderer?.domElement.removeEventListener('webglcontextlost', contextLost);
      renderer?.dispose();
      renderer?.domElement.remove();
      element.removeEventListener('pointermove', move);
      element.removeEventListener('pointerleave', leave);
      element.removeEventListener('pointerdown', pointerDown);
      element.removeEventListener('pointerup', pointerUp);
    };
  }, []);

  return <div ref={host} className={`brain-canvas ${ready ? 'is-ready' : ''}`} />;
}
