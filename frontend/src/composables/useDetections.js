/**
 * Detection filtering composable — applies a confidence threshold to
 * nodule detections and exposes derived stats (count, max score).
 *
 * @module useDetections
 */
import { ref, computed } from 'vue'

/**
 * Filter and sort detections from a reactive study result by confidence score.
 *
 * @param {import('vue').Ref<Object|null>} result - Reactive ref to the study result
 *   object. Expected to have a `detections` array with `{ score, x1, y1, x2, y2 }` items.
 * @returns {{
 *   threshold: import('vue').Ref<number>,
 *   filtered: import('vue').ComputedRef<Array<Object>>,
 *   count: import('vue').ComputedRef<number>,
 *   maxScore: import('vue').ComputedRef<number>
 * }}
 */
export function useDetections(result) {
  /** @type {import('vue').Ref<number>} Minimum confidence score (0–1) for a detection to be shown. */
  const threshold = ref(0.3)

  /** Detections above the threshold, sorted by score descending. */
  const filtered = computed(() => {
    if (!result.value?.detections) return []
    return result.value.detections
      .filter(d => d.score >= threshold.value)
      .sort((a, b) => b.score - a.score)
  })

  /** Number of detections passing the threshold. */
  const count = computed(() => filtered.value.length)

  /** Highest confidence score among filtered detections (0 if none). */
  const maxScore = computed(() => {
    if (filtered.value.length === 0) return 0
    return Math.max(...filtered.value.map(d => d.score))
  })

  return { threshold, filtered, count, maxScore }
}
