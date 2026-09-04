import { useEffect, useRef, useState } from 'react';

/**
 * Interpolate a number toward a new value over `duration` milliseconds.
 *
 * Three rules keep this from being annoying rather than informative:
 *
 *   - the first value is never animated, because a count-up from zero on page
 *     load is decoration, not feedback;
 *   - under `prefers-reduced-motion` the raw value is returned immediately;
 *   - the final frame snaps to the exact target, so a readout is never left
 *     one off by floating-point drift.
 *
 * @param {number} value
 * @param {number} [duration] milliseconds
 * @returns {number}
 */
export function useAnimatedNumber(value, duration = 640) {
  const target = Number.isFinite(value) ? value : 0;
  const [display, setDisplay] = useState(target);
  const fromRef = useRef(target);
  const frameRef = useRef(0);
  const primedRef = useRef(false);

  useEffect(() => {
    // First run: adopt the value without animating it.
    if (!primedRef.current) {
      primedRef.current = true;
      fromRef.current = target;
      setDisplay(target);
      return undefined;
    }

    const reduced = typeof window !== 'undefined'
      && window.matchMedia
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (reduced || duration <= 0) {
      fromRef.current = target;
      setDisplay(target);
      return undefined;
    }

    const from = fromRef.current;
    if (from === target) return undefined;

    const start = performance.now();

    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      // easeOutExpo: most of the distance is covered early, so the number
      // reads as "settling" rather than "counting".
      const eased = t >= 1 ? 1 : 1 - Math.pow(2, -10 * t);
      const next = from + (target - from) * eased;
      fromRef.current = next;

      if (t >= 1) {
        fromRef.current = target;
        setDisplay(target);
        return;
      }
      setDisplay(next);
      frameRef.current = requestAnimationFrame(step);
    };

    frameRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frameRef.current);
  }, [target, duration]);

  return display;
}
