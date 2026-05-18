<script setup>
/**
 * DetectionList — renders a vertical list of nodule detection cards.
 * Each card shows risk level (color-coded), confidence score bar, and bounding-box coordinates.
 */
import { useI18n } from '../i18n/index.js'

const { t } = useI18n()
const props = defineProps({ detections: Array })

function scoreColor(score) {
  if (score >= 0.8) return 'text-red-400 bg-red-500/10'
  if (score >= 0.5) return 'text-amber-400 bg-amber-500/10'
  return 'text-gray-400 bg-gray-500/10'
}

function riskLabel(score) {
  if (score >= 0.8) return t('detection.high')
  if (score >= 0.5) return t('detection.medium')
  return t('detection.low')
}
</script>

<template>
  <div class="space-y-2">
    <h3 class="text-sm font-semibold text-[var(--text-primary)] uppercase tracking-wider">{{ t('detection.title') }}</h3>
    <div v-if="detections.length === 0" class="text-[var(--text-secondary)] text-sm py-4 text-center">
      {{ t('detection.empty') }}
    </div>
    <div
      v-for="(d, i) in detections" :key="i"
      class="bg-[var(--bg-card)] rounded-lg p-3 border border-[var(--border)] hover:border-[var(--text-secondary)] transition-colors"
    >
      <div class="flex items-center justify-between mb-2">
        <div class="flex items-center gap-2">
          <span class="text-[var(--text-primary)] font-medium text-sm">{{ t('analyze.nodule') }} #{{ i + 1 }}</span>
          <span :class="scoreColor(d.score)" class="text-xs px-2 py-0.5 rounded-full font-semibold">
            {{ riskLabel(d.score) }}
          </span>
        </div>
        <span class="text-emerald-400 font-mono font-bold text-sm">{{ (d.score * 100).toFixed(1) }}%</span>
      </div>

      <!-- Score bar -->
      <div class="w-full bg-[var(--border)] rounded-full h-1.5 mb-2">
        <div
          class="h-1.5 rounded-full transition-all duration-300"
          :class="d.score >= 0.8 ? 'bg-red-400' : d.score >= 0.5 ? 'bg-amber-400' : 'bg-gray-400'"
          :style="{ width: `${d.score * 100}%` }"
        />
      </div>

      <!-- Coordinates -->
      <div class="text-xs text-[var(--text-secondary)] font-mono">
        {{ t('detection.pos') }}: ({{ Math.round(d.x1) }}, {{ Math.round(d.y1) }}) - ({{ Math.round(d.x2) }}, {{ Math.round(d.y2) }})
        <span class="ml-2">{{ Math.round(d.x2 - d.x1) }}x{{ Math.round(d.y2 - d.y1) }}px</span>
      </div>
    </div>
  </div>
</template>
