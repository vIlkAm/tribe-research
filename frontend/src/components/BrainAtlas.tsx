import { useEffect, useId, useRef, useState, type PointerEvent } from 'react';
import type { Analysis } from '../data/analysis.types';
import { frameAt, isGap, sampleAt, signalText } from '../lib/analysis';
import { analysisAssetUrl } from '../lib/assets';
import './contract-widgets.css';

interface Props { analysis: Analysis; timeMs: number; onHover?: (key: string | null) => void; resolveAsset?: (relative: string) => string | null; }
type LoadState = 'loading' | 'ready' | 'error';

export default function BrainAtlas({ analysis, timeMs, onHover, resolveAsset = analysisAssetUrl }: Props) {
  const clipId = useId().replaceAll(':', '');
  // Raw vertex maps must never be displayed by this normal-mode component.
  const sprite = analysis.assets.brain_map?.mode === 'proxy' ? analysis.assets.brain_map : undefined;
  const regionMap = analysis.assets.region_map;
  const file = sprite ? resolveAsset(sprite.src) : null;
  const idmapFile = sprite && regionMap ? resolveAsset(regionMap.idmap_src) : null;
  const legendFile = sprite && regionMap ? resolveAsset(regionMap.legend_src) : null;
  const [spriteLoad, setSpriteLoad] = useState<{ src: string | null; state: LoadState }>({ src: null, state: 'loading' });
  const [mapLoad, setMapLoad] = useState<{ src: string | null; state: LoadState }>({ src: null, state: 'loading' });
  const [legendError, setLegendError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const pixels = useRef<ImageData | null>(null);
  const frame = sprite ? frameAt(sprite, timeMs) : -1;
  const gap = isGap(analysis, timeMs) || frame < 0;
  const imageState = spriteLoad.src === file ? spriteLoad.state : 'loading';
  const mapState = mapLoad.src === idmapFile ? mapLoad.state : 'loading';
  const channel = analysis.channels.find(candidate => candidate.key === hovered);

  useEffect(() => {
    if (!file) return;
    let active = true;
    const image = new Image();
    image.onload = () => { if (active) setSpriteLoad({ src: file, state: 'ready' }); };
    image.onerror = () => { if (active) setSpriteLoad({ src: file, state: 'error' }); };
    image.src = file;
    return () => { active = false; image.onload = null; image.onerror = null; };
  }, [file]);

  useEffect(() => {
    pixels.current = null;
    if (!idmapFile || !sprite) return;
    let active = true;
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => {
      if (!active) return;
      try {
        if (image.naturalWidth !== sprite.tile_w || image.naturalHeight !== sprite.tile_h) throw new Error('Region map geometry does not match the brain tile.');
        const canvas = document.createElement('canvas');
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const context = canvas.getContext('2d', { willReadFrequently: true });
        if (!context) throw new Error('Region map pixels unavailable.');
        context.imageSmoothingEnabled = false;
        context.drawImage(image, 0, 0);
        pixels.current = context.getImageData(0, 0, canvas.width, canvas.height);
        setMapLoad({ src: idmapFile, state: 'ready' });
      } catch { setMapLoad({ src: idmapFile, state: 'error' }); }
    };
    image.onerror = () => { if (active) setMapLoad({ src: idmapFile, state: 'error' }); };
    image.src = idmapFile;
    return () => { active = false; image.onload = null; image.onerror = null; };
  }, [idmapFile, sprite?.tile_w, sprite?.tile_h]);

  useEffect(() => { if (gap) setHovered(null); }, [gap]);
  useEffect(() => { setHovered(null); }, [idmapFile]);
  useEffect(() => { onHover?.(hovered); return () => onHover?.(null); }, [hovered, onHover]);

  function inspect(event: PointerEvent<SVGSVGElement>) {
    if (!sprite || !regionMap || !pixels.current || mapState !== 'ready' || imageState !== 'ready' || gap) { setHovered(null); return; }
    const bounds = event.currentTarget.getBoundingClientRect();
    const scale = Math.min(bounds.width / sprite.tile_w, bounds.height / sprite.tile_h);
    const x = Math.floor((event.clientX - bounds.left - (bounds.width - sprite.tile_w * scale) / 2) / scale);
    const y = Math.floor((event.clientY - bounds.top - (bounds.height - sprite.tile_h * scale) / 2) / scale);
    if (x < 0 || y < 0 || x >= sprite.tile_w || y >= sprite.tile_h) { setHovered(null); return; }
    const offset = (y * pixels.current.width + x) * 4;
    const rgb = `#${Array.from(pixels.current.data.slice(offset, offset + 3), value => value.toString(16).padStart(2, '0')).join('')}`;
    setHovered(rgb === '#000000' ? null : regionMap.ids[rgb] ?? null);
  }

  if (!sprite || !file) return <div className="atlas-empty">Proxy cortical images unavailable for this analysis. You can still explore its signal timeline.</div>;
  const value = channel && !gap ? sampleAt(channel, timeMs, analysis.duration_ms) : null;
  return <div className="atlas-wrap contract-atlas">
    <div className="contract-atlas-surface">
      <svg className="contract-atlas-canvas" viewBox={`0 0 ${sprite.tile_w} ${sprite.tile_h}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label="Four proxy cortical views supplied by tribe-research" onPointerMove={inspect} onPointerLeave={() => setHovered(null)}>
        <defs><clipPath id={clipId}><rect width={sprite.tile_w} height={sprite.tile_h} /></clipPath></defs>
        {!gap && imageState === 'ready' && <image href={file!} x={-(frame % sprite.cols) * sprite.tile_w} y={-Math.floor(frame / sprite.cols) * sprite.tile_h} width={sprite.cols * sprite.tile_w} height={sprite.rows * sprite.tile_h} preserveAspectRatio="none" clipPath={`url(#${clipId})`} />}
      </svg>
      {(gap || imageState !== 'ready') && <span className="contract-atlas-status" role="status">{gap ? 'No prediction at this moment' : imageState === 'error' ? 'Cortical image unavailable' : 'Loading cortical image…'}</span>}
      {channel && !gap && imageState === 'ready' && mapState === 'ready' && <div className="contract-atlas-tooltip"><strong style={{ color: channel.color }}>{channel.label} <span>{value === null ? 'No prediction' : `${signalText(value)} z`}</span></strong><small>Relative to this clip</small><p>{channel.basis.regions_text}</p><p>{channel.copy.tooltip}</p><small>{channel.research_status === 'validated' ? 'Validated signal' : 'Research proxy'} · {channel.confidence} · {channel.source_model}</small></div>}
    </div>
    <div className="contract-atlas-tools"><span>{!regionMap || !idmapFile ? 'Region hover unavailable' : mapState === 'error' ? 'Region hover unavailable: the region map could not be read.' : mapState === 'loading' ? 'Loading region map…' : 'Hover over a region to inspect its proxy'}</span>
      {legendFile && <details className="contract-atlas-legend"><summary>Atlas legend</summary><div>{legendError === legendFile ? <p role="status">Atlas legend unavailable.</p> : <img src={legendFile} alt="Supplied channel colors, directions and within-clip response scale" onError={() => setLegendError(legendFile)} />}</div></details>}
    </div>
    <p className="contract-atlas-caption">TRIBE v2 prediction · average subject · research proxy, not measured brain activity</p>
  </div>;
}
