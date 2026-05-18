<script setup>
/**
 * StatsBar — fetches and displays aggregate CXR statistics on mount
 * (total studies, completed, with nodules, average inference time).
 */
import { ref, onMounted, watch } from 'vue'
import { getStats } from '../lib/api.js'
import { useI18n } from '../i18n/index.js'

const { t } = useI18n()
const props = defineProps({ refreshKey: { type: Number, default: 0 } })
const stats = ref(null)
const fetchError = ref(false)

async function refresh() {
  try {
    stats.value = await getStats()
    fetchError.value = false
  } catch {
    fetchError.value = true
  }
}

onMounted(refresh)
watch(() => props.refreshKey, refresh)
</script>

<template>
  <div v-if="stats" class="flex gap-6 text-sm">
    <div class="text-center">
      <div class="text-xl font-bold text-[var(--text-primary)]">{{ stats.total_studies }}</div>
      <div class="text-[var(--text-secondary)] text-xs">{{ t('stats.total') }}</div>
    </div>
    <div class="text-center">
      <div class="text-xl font-bold text-emerald-400">{{ stats.completed }}</div>
      <div class="text-[var(--text-secondary)] text-xs">{{ t('stats.analyzed') }}</div>
    </div>
    <div class="text-center">
      <div class="text-xl font-bold text-red-400">{{ stats.with_nodules }}</div>
      <div class="text-[var(--text-secondary)] text-xs">{{ t('stats.withNodules') }}</div>
    </div>
    <div v-if="stats.avg_inference_ms" class="text-center">
      <div class="text-xl font-bold text-blue-400">{{ Math.round(stats.avg_inference_ms) }}ms</div>
      <div class="text-[var(--text-secondary)] text-xs">{{ t('stats.avgTime') }}</div>
    </div>
  </div>
</template>
