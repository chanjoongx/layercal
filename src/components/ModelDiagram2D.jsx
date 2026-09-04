import React, { useId, useMemo } from 'react';

/**
 * Isometric SVG rendering of a Scene.
 *
 * This is not a degraded mode dressed up as a feature. It is what renders
 * when WebGL2 is unavailable, what renders while the WebGL module is still
 * being fetched, what the AI Advisor uses to preview a proposal, and what the
 * server renders. It reads the same `Scene` the GPU renderer does, so the two
 * can never show different models.
 *
 * Projection: the standard 2:1 isometric, with +z running right-and-down and
 * +x left-and-down, so a layer stack laid out along +z reads left to right.
 */

const SCALE = 62;
const PADDING = 26;

const project = (x, y, z) => ({
  sx: (z - x) * 0.866 * SCALE,
  sy: ((z + x) * 0.5 - y) * SCALE,
});

/** Lighten (t > 0) or darken (t < 0) a #rrggbb colour. */
function shade(hex, t) {
  const value = parseInt(hex.slice(1), 16);
  const channel = (shift) => {
    const c = (value >> shift) & 0xff;
    const next = t >= 0 ? c + (255 - c) * t : c * (1 + t);
    return Math.max(0, Math.min(255, Math.round(next)));
  };
  return `rgb(${channel(16)}, ${channel(8)}, ${channel(0)})`;
}

const points = (list) => list.map(p => `${p.sx.toFixed(2)},${p.sy.toFixed(2)}`).join(' ');

/**
 * The three faces of an axis-aligned box that face this viewpoint: the top,
 * the -z face (left) and the -x face (right).
 */
function boxFaces(node) {
  const { center, extent } = node;
  const hw = extent.w / 2;
  const hh = extent.h / 2;
  const hd = extent.d / 2;
  const at = (dx, dy, dz) => project(center.x + dx, center.y + dy, center.z + dz);

  return {
    top: [at(-hw, hh, -hd), at(hw, hh, -hd), at(hw, hh, hd), at(-hw, hh, hd)],
    left: [at(-hw, hh, -hd), at(hw, hh, -hd), at(hw, -hh, -hd), at(-hw, -hh, -hd)],
    right: [at(-hw, hh, -hd), at(-hw, hh, hd), at(-hw, -hh, hd), at(-hw, -hh, -hd)],
  };
}

/** A tapered quad along the flow axis, drawn behind the boxes. */
function linkShape(scene, link) {
  const a = scene.nodes[link.from];
  const b = scene.nodes[link.to];
  const z0 = a.center.z + a.extent.d / 2;
  const z1 = b.center.z - b.extent.d / 2;
  const w0 = Math.max(0.04, link.width0);
  const w1 = Math.max(0.04, link.width1);

  return [
    project(a.center.x, a.center.y + w0, z0),
    project(b.center.x, b.center.y + w1, z1),
    project(b.center.x, b.center.y - w1, z1),
    project(a.center.x, a.center.y - w0, z0),
  ];
}

/**
 * @param {{
 *   scene: object,
 *   isDarkMode?: boolean,
 *   selectedId?: string|null,
 *   onSelect?: (id: string|null) => void,
 *   showLabels?: boolean,
 *   className?: string,
 *   label?: string,
 *   emptyMessage?: string,
 * }} props
 */
export default function ModelDiagram2D({
  scene,
  isDarkMode = false,
  selectedId = null,
  onSelect,
  showLabels = false,
  className = '',
  label,
  emptyMessage,
}) {
  const geometry = useMemo(() => {
    if (!scene || scene.nodes.length === 0) return null;

    const nodes = scene.nodes.map(node => ({
      node,
      faces: boxFaces(node),
      anchor: project(node.center.x, node.center.y + node.extent.h / 2, node.center.z),
    }));
    const links = scene.links.map(link => ({ link, shape: linkShape(scene, link) }));

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const entry of nodes) {
      for (const face of Object.values(entry.faces)) {
        for (const p of face) {
          if (p.sx < minX) minX = p.sx;
          if (p.sx > maxX) maxX = p.sx;
          if (p.sy < minY) minY = p.sy;
          if (p.sy > maxY) maxY = p.sy;
        }
      }
    }

    return {
      nodes,
      links,
      viewBox: [
        minX - PADDING,
        minY - PADDING,
        (maxX - minX) + PADDING * 2,
        (maxY - minY) + PADDING * 2,
      ],
    };
  }, [scene]);

  // Two of these can be on the page at once - the WebGL fallback and the
  // advisor's proposal preview - and fixed gradient ids would collide, so the
  // second instance would paint with the first one's colours.
  const uid = useId().replace(/:/g, '');
  const gradientId = (i) => `layercal-flow-${uid}-${i}`;

  const muted = isDarkMode ? '#64748b' : '#94a3b8';

  if (!geometry) {
    return (
      <div className={`flex items-center justify-center text-sm ${className}`} style={{ color: muted }}>
        {emptyMessage || 'Nothing to draw yet.'}
      </div>
    );
  }

  return (
    <svg
      className={className}
      viewBox={geometry.viewBox.join(' ')}
      role="img"
      aria-label={label || 'Model architecture diagram'}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        {/* One gradient per link, so a connection visibly carries the colour of
            the layer it comes from into the layer it feeds. */}
        {geometry.links.map(({ link }, i) => {
          const from = scene.nodes[link.from];
          const to = scene.nodes[link.to];
          return (
            <linearGradient key={i} id={gradientId(i)} x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={isDarkMode ? from.paint.hexDark : from.paint.hex} />
              <stop offset="100%" stopColor={isDarkMode ? to.paint.hexDark : to.paint.hex} />
            </linearGradient>
          );
        })}
      </defs>

      <g>
        {geometry.links.map(({ link, shape }, i) => (
          <polygon
            key={`link-${i}`}
            points={points(shape)}
            fill={link.broken ? 'none' : `url(#${gradientId(i)})`}
            fillOpacity={0.42}
            stroke={link.broken ? (isDarkMode ? '#fb923c' : '#ea580c') : 'none'}
            strokeWidth={link.broken ? 1.6 : 0}
            strokeDasharray={link.broken ? '5 4' : undefined}
          />
        ))}
      </g>

      <g>
        {geometry.nodes.map(({ node, faces, anchor }) => {
          const hex = isDarkMode ? node.paint.hexDark : node.paint.hex;
          const selected = node.id === selectedId;
          const interactive = typeof onSelect === 'function';

          return (
            <g
              key={node.id}
              onClick={interactive ? () => onSelect(node.id) : undefined}
              style={interactive ? { cursor: 'pointer' } : undefined}
              opacity={node.annotation ? 0.85 : 1}
            >
              <title>
                {`${node.name} — ${node.shape.label} — ${node.params.toLocaleString()} parameters`
                  + (node.warning ? ' — dimension mismatch' : '')}
              </title>
              <polygon points={points(faces.right)} fill={shade(hex, -0.16)} />
              <polygon points={points(faces.left)} fill={hex} />
              <polygon points={points(faces.top)} fill={shade(hex, 0.14)} />
              <polygon
                points={points(faces.top)}
                fill="none"
                stroke={selected ? (isDarkMode ? '#e9d5ff' : '#4c1d95') : shade(hex, 0.3)}
                strokeWidth={selected ? 2 : 0.6}
              />
              {node.warning && (
                <polygon
                  points={points(faces.left)}
                  fill="none"
                  stroke={isDarkMode ? '#fb923c' : '#c2410c'}
                  strokeWidth={2}
                  strokeDasharray="4 3"
                />
              )}
              {showLabels && !node.annotation && (
                <text
                  x={anchor.sx}
                  y={anchor.sy - 8}
                  textAnchor="middle"
                  fontSize={11}
                  fill={isDarkMode ? '#cbd5e1' : '#334155'}
                  style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}
                >
                  {node.name}
                </text>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
}
