import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Play, Pause, Maximize2, Tag, Camera, Boxes, Cpu, Layers3 } from 'lucide-react';
import { buildScene, describeScene } from '@/viz/sceneGraph';
import { LAYER_FAMILIES, FAMILY_LABELS } from '@/viz/palette';
import { formatNumber } from '@/config/layerTypes';
import ModelDiagram2D from '@/components/ModelDiagram2D';

/**
 * The live architecture panel.
 *
 * The WebGL module is pulled in by a dynamic import inside an effect, never a
 * static import and never React.lazy: `renderToStaticMarkup` cannot suspend, so
 * lazy would break server rendering, and an effect simply never runs there. Until
 * the module resolves — and permanently, if the browser has no WebGL2 — the
 * isometric SVG stands in, so the panel is never blank and the layout never
 * shifts under the user.
 */

const prefersReducedMotion = () => {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
};

export default function ModelViewer({
  layers,
  layerTypes,
  issues,
  isDarkMode,
  t,
  selectedId,
  onSelect,
  onFocusLayer,
  captureRef,
}) {
  const canvasRef = useRef(null);
  const engineRef = useRef(null);
  const labelRefs = useRef(new Map());

  // Bumped to tear the engine down and build a fresh one. The only thing that
  // does that is a restored GL context: every program, buffer and vertex array
  // from the old context is gone, and rebuilding them in place would mean a
  // second construction path nothing else exercises.
  const [engineEpoch, setEngineEpoch] = useState(0);
  const [engineReady, setEngineReady] = useState(false);
  const [unsupported, setUnsupported] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [hoveredId, setHoveredId] = useState(null);
  const [reducedMotion, setReducedMotion] = useState(prefersReducedMotion);

  const scene = useMemo(
    () => buildScene(layers, layerTypes, issues),
    [layers, layerTypes, issues]
  );

  // The scene lives in a ref as well, so the imperative callbacks handed to the
  // engine never close over a stale copy.
  const sceneRef = useRef(scene);
  sceneRef.current = scene;

  // Read once when a fresh engine starts, so a context restore does not lose
  // whichever layer the user had selected.
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId || null;

  const activeNode = useMemo(() => {
    const id = hoveredId || selectedId;
    return id ? scene.nodes.find(n => n.id === id) || null : null;
  }, [hoveredId, selectedId, scene]);

  const families = useMemo(() => {
    const present = new Set(scene.nodes.map(n => n.paint.family));
    return LAYER_FAMILIES.filter(f => present.has(f));
  }, [scene]);

  // ── label positioning ─────────────────────────────
  // Called once per rendered frame. Writing transforms straight onto the DOM
  // nodes keeps this off React's critical path; a setState here would re-render
  // the whole panel sixty times a second.
  const handleFrame = useCallback((labels) => {
    const map = labelRefs.current;
    const seen = new Set();
    for (const item of labels) {
      const el = map.get(item.id);
      seen.add(item.id);
      if (!el) continue;
      if (item.opacity <= 0.02) {
        el.style.visibility = 'hidden';
        continue;
      }
      el.style.visibility = 'visible';
      el.style.opacity = String(item.opacity);
      el.style.transform = `translate3d(${item.x}px, ${item.y}px, 0) translate(-50%, -100%)`;
    }
    for (const [id, el] of map) {
      if (!seen.has(id) && el) el.style.visibility = 'hidden';
    }
  }, []);

  const handleHover = useCallback((id) => setHoveredId(id), []);

  const handlePick = useCallback((id) => {
    onSelect?.(id);
    if (id) onFocusLayer?.(id);
  }, [onSelect, onFocusLayer]);

  const handleContextLost = useCallback(() => setEngineReady(false), []);
  const handleContextRestored = useCallback(() => setEngineEpoch(e => e + 1), []);

  // The engine is created once and keeps whatever callbacks it was handed. They
  // are stable today, but routing through a ref means a future change that
  // makes one unstable cannot silently leave the engine calling a stale
  // closure - a failure that would show up as the canvas quietly going inert.
  const handlers = useRef(null);
  handlers.current = {
    onHover: handleHover,
    onSelect: handlePick,
    onFrame: handleFrame,
    onContextLost: handleContextLost,
    onContextRestored: handleContextRestored,
  };

  // ── engine lifecycle ──────────────────────────────
  useEffect(() => {
    let cancelled = false;
    let engine = null;

    import('@/viz/renderer')
      .then(({ createRenderer }) => {
        if (cancelled || !canvasRef.current) return;
        engine = createRenderer(canvasRef.current, {
          isDarkMode,
          motion: !prefersReducedMotion(),
          onHover: (id) => handlers.current.onHover(id),
          onSelect: (id) => handlers.current.onSelect(id),
          onFrame: (items) => handlers.current.onFrame(items),
          onContextLost: () => handlers.current.onContextLost(),
          onContextRestored: () => handlers.current.onContextRestored(),
        });
        if (!engine) {
          setUnsupported(true);
          return;
        }
        engineRef.current = engine;
        engine.setScene(sceneRef.current, { keepCamera: false });
        engine.setSelection(selectedRef.current, null);
        engine.start();
        setEngineReady(true);
      })
      .catch(() => {
        if (!cancelled) setUnsupported(true);
      });

    return () => {
      cancelled = true;
      engineRef.current = null;
      setEngineReady(false);
      engine?.dispose();
    };
    // Re-runs only when the epoch changes. Theme, scene and selection are
    // pushed through their own effects rather than by tearing down the context.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engineEpoch]);

  // Adding or removing a layer changes the shape of the model, so the camera
  // refits; editing a parameter does not, because refitting on every keystroke
  // would make the view lurch while someone types. The orbit angle is kept -
  // only the distance and the centre move - so this reads as the frame
  // following the model rather than the camera being taken away.
  const nodeCountRef = useRef(scene.nodes.length);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.setScene(scene);
    if (scene.nodes.length !== nodeCountRef.current) {
      nodeCountRef.current = scene.nodes.length;
      engine.frameAll();
    }
  }, [scene]);

  useEffect(() => {
    engineRef.current?.setTheme(isDarkMode);
  }, [isDarkMode]);

  useEffect(() => {
    engineRef.current?.setSelection(selectedId || null, hoveredId);
  }, [selectedId, hoveredId]);

  useEffect(() => {
    engineRef.current?.setPlaying(playing);
  }, [playing]);

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined;
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const apply = () => {
      setReducedMotion(query.matches);
      engineRef.current?.setMotion(!query.matches);
    };
    apply();
    query.addEventListener?.('change', apply);
    return () => query.removeEventListener?.('change', apply);
  }, [engineReady]);

  // The PNG export renders the page through html2canvas, which reads the canvas
  // rather than the GPU. Without a synchronous frame first, a paused or
  // offscreen viewer exports as an empty rectangle.
  useEffect(() => {
    if (!captureRef) return undefined;
    captureRef.current = () => engineRef.current?.renderNow();
    return () => { captureRef.current = null; };
  }, [captureRef]);

  const handleReset = useCallback(() => engineRef.current?.frameAll(), []);

  const handleSnapshot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    engineRef.current?.renderNow();
    try {
      const url = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.href = url;
      link.download = 'layercal-architecture.png';
      link.click();
    } catch {
      // A tainted canvas cannot happen here (nothing external is drawn), but a
      // browser can still refuse the read in a private context.
    }
  }, []);

  const showCanvas = engineReady && !unsupported;
  const ariaLabel = describeScene(scene, t);
  const isEmpty = scene.nodes.length === 0;

  const surface = isDarkMode ? 'border-white/10' : 'border-slate-200';
  const chip = isDarkMode
    ? 'bg-black/45 text-slate-200 border-white/10'
    : 'bg-white/80 text-slate-700 border-slate-200';
  const button = isDarkMode
    ? 'bg-black/45 text-slate-200 border-white/10 hover:bg-black/70 hover:text-white'
    : 'bg-white/85 text-slate-600 border-slate-200 hover:bg-white hover:text-slate-900';

  return (
    <section
      className={`viz-surface relative overflow-hidden rounded-2xl border shadow-sm ${surface}`}
      aria-labelledby="viz-heading"
    >
      <h2 id="viz-heading" className="sr-only">
        {t.vizTitle || 'Live architecture'}
      </h2>

      <div className="relative aspect-[4/3] sm:aspect-[2/1] lg:aspect-[21/9] min-h-[280px] max-h-[560px]">
        {/* The canvas is present from the first paint so its size is stable,
            and only becomes visible once the engine has drawn into it. */}
        <canvas
          ref={canvasRef}
          tabIndex={showCanvas ? 0 : -1}
          // While the SVG fallback is showing it carries the description; a
          // second labelled image here would have a screen reader announce the
          // same model twice.
          {...(showCanvas
            ? { role: 'img', 'aria-label': ariaLabel }
            : { 'aria-hidden': true })}
          className="absolute inset-0 h-full w-full rounded-2xl outline-none"
          style={{
            touchAction: 'none',
            cursor: 'grab',
            opacity: showCanvas ? 1 : 0,
            transition: 'opacity 320ms cubic-bezier(.16,1,.3,1)',
          }}
        />

        {!showCanvas && (
          <div className="absolute inset-0 flex items-center justify-center p-6">
            <ModelDiagram2D
              scene={scene}
              isDarkMode={isDarkMode}
              selectedId={selectedId}
              onSelect={handlePick}
              showLabels
              className="h-full w-full"
              label={ariaLabel}
              emptyMessage={t.vizEmpty || 'Add a layer to see it here'}
            />
          </div>
        )}

        {isEmpty && showCanvas && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <p className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
              {t.vizEmpty || 'Add a layer to see it here'}
            </p>
          </div>
        )}

        {/* Overlay chrome. The container ignores pointer events so the canvas
            stays draggable; each control opts back in. */}
        <div className="pointer-events-none absolute inset-0">
          {/* Floating layer labels */}
          {showLabels && showCanvas && (
            <div className="absolute inset-0 overflow-hidden">
              {scene.nodes.filter(n => !n.annotation).map(node => (
                <span
                  key={node.id}
                  ref={(el) => {
                    if (el) labelRefs.current.set(node.id, el);
                    else labelRefs.current.delete(node.id);
                  }}
                  className={`absolute left-0 top-0 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[10px] font-medium tracking-tight ${chip}`}
                  style={{ visibility: 'hidden', willChange: 'transform' }}
                  aria-hidden="true"
                >
                  {node.name}
                </span>
              ))}
            </div>
          )}

          {/* Stats. The width cap reserves the toolbar's corner: without it the
              chips wrap onto a second line on a phone and land under the
              buttons, which is unreachable rather than merely untidy. */}
          <div className="absolute left-3 top-3 flex max-w-[calc(100%-9rem)] flex-wrap gap-1.5">
            <Stat icon={Boxes} label={t.totalParameters} value={formatNumber(scene.totals.params)} className={chip} />
            <Stat icon={Cpu} label={t.totalFLOPs || 'FLOPs'} value={formatNumber(scene.totals.flops)} className={`hidden sm:flex ${chip}`} />
            <Stat icon={Layers3} label={t.numberOfLayers} value={String(scene.totals.depth)} className={`hidden sm:flex ${chip}`} />
          </div>

          {/* Toolbar */}
          <div className="pointer-events-auto absolute right-3 top-3 flex gap-1.5">
            {showCanvas && !reducedMotion && (
              <ToolButton
                className={button}
                onClick={() => setPlaying(p => !p)}
                label={playing ? (t.vizPause || 'Pause flow') : (t.vizPlay || 'Play flow')}
              >
                {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
              </ToolButton>
            )}
            <ToolButton
              className={button}
              onClick={() => setShowLabels(v => !v)}
              label={t.vizLabels || 'Toggle labels'}
              pressed={showLabels}
            >
              <Tag className="h-3.5 w-3.5" />
            </ToolButton>
            {showCanvas && (
              <>
                <ToolButton className={button} onClick={handleReset} label={t.vizReset || 'Reset view'}>
                  <Maximize2 className="h-3.5 w-3.5" />
                </ToolButton>
                <ToolButton className={button} onClick={handleSnapshot} label={t.vizSnapshot || 'Save as PNG'}>
                  <Camera className="h-3.5 w-3.5" />
                </ToolButton>
              </>
            )}
          </div>

          {/* Legend */}
          {families.length > 0 && (
            <ul className={`absolute bottom-3 left-3 hidden max-w-[60%] flex-wrap gap-x-3 gap-y-1 rounded-lg border px-2 py-1.5 sm:flex ${chip}`}>
              {families.map(family => (
                <li key={family} className="flex items-center gap-1.5 text-[10px]">
                  <span
                    className="h-2 w-2 rounded-[3px]"
                    style={{ background: familySwatch(scene, family, isDarkMode) }}
                    aria-hidden="true"
                  />
                  {t[`fam_${family}`] || FAMILY_LABELS[family]}
                </li>
              ))}
            </ul>
          )}

          {/* Hover / selection readout */}
          {activeNode && (
            <div
              aria-live="polite"
              className={`absolute bottom-3 right-3 max-w-[16rem] rounded-lg border px-3 py-2 text-xs ${chip}`}
            >
              <div className="flex items-center gap-1.5 font-semibold">
                <span
                  className="h-2.5 w-2.5 rounded-[3px]"
                  style={{ background: isDarkMode ? activeNode.paint.hexDark : activeNode.paint.hex }}
                  aria-hidden="true"
                />
                {activeNode.name}
              </div>
              <dl className="mt-1 space-y-0.5 tabular-nums">
                <Row label={t.vizShape || 'Shape'} value={activeNode.shape.label} />
                <Row label={t.parameters} value={activeNode.params.toLocaleString()} />
                <Row label={t.vizShare || 'Share'} value={`${(activeNode.paramShare * 100).toFixed(1)}%`} />
              </dl>
              {activeNode.warning && (
                <p className={isDarkMode ? 'mt-1 text-amber-300' : 'mt-1 text-amber-700'}>
                  {(t.dimMismatch || 'Set {field} to {n} to match the previous layer')
                    .replace('{field}', activeNode.warningField || '')
                    .replace('{n}', String(activeNode.warningExpected))}
                </p>
              )}
            </div>
          )}

          {unsupported && (
            <p className={`absolute bottom-3 right-3 rounded-lg border px-2 py-1 text-[11px] ${chip}`}>
              {t.vizFallbackNote || 'WebGL is unavailable here, so this is a static diagram.'}
            </p>
          )}
        </div>
      </div>

      {/* The full content of the visualisation, for assistive technology. */}
      <ul className="sr-only">
        {scene.nodes.map(node => (
          <li key={node.id}>
            {`${node.index + 1}. ${node.name}, ${node.shape.label}, ${node.params.toLocaleString()} ${t.parameters}`}
          </li>
        ))}
      </ul>

      <p className={`px-3 pb-2 pt-1 text-[11px] ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
        {t.vizHint || 'Drag to orbit, scroll to zoom, click a layer to focus it.'}
      </p>
    </section>
  );
}

function familySwatch(scene, family, isDarkMode) {
  const node = scene.nodes.find(n => n.paint.family === family);
  if (!node) return 'transparent';
  return isDarkMode ? node.paint.hexDark : node.paint.hex;
}

function Stat({ icon: Icon, label, value, className }) {
  return (
    <span className={`flex items-center gap-1.5 rounded-lg border px-2 py-1 text-[11px] tabular-nums ${className}`}>
      <Icon className="h-3 w-3 opacity-70" aria-hidden="true" />
      <span className="opacity-70">{label}</span>
      <strong className="font-semibold">{value}</strong>
    </span>
  );
}

function ToolButton({ children, label, onClick, className, pressed }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={pressed}
      className={`rounded-lg border p-1.5 transition-colors ${className}`}
    >
      {children}
    </button>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="opacity-70">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}
