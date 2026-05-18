<script setup>
/**
 * ValidationStatsView — displays aggregate validation statistics and provides
 * dataset export buttons (CSV/JSON). Fetches stats on mount from the backend.
 */
import { ref, onMounted, computed } from 'vue'
import { getValidationStats, getExportUrl } from '../lib/api.js'
import { useI18n } from '../i18n/index.js'

const { t } = useI18n()
const stats = ref(null)
const loading = ref(true)
const error = ref(null)

onMounted(async () => {
  try {
    stats.value = await getValidationStats()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

const accuracy = computed(() => {
  if (!stats.value?.accuracy) return null
  return (stats.value.accuracy * 100).toFixed(1)
})

function downloadExport(format) {
  window.open(getExportUrl(format), '_blank')
}
</script>

<template>
  <div class="max-w-[900px] mx-auto px-4 py-6">
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-lg font-bold text-[var(--text-primary)]">{{ t('validationStats.title') }}</h2>
      <div class="flex gap-2">
        <button
          @click="downloadExport('csv')"
          class="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-4 py-1.5 rounded-lg transition-colors"
        >{{ t('validationStats.exportCsv') }}</button>
        <button
          @click="downloadExport('json')"
          class="bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-1.5 rounded-lg transition-colors"
        >{{ t('validationStats.exportJson') }}</button>
      </div>
    </div>

    <div v-if="loading" class="text-[var(--text-secondary)] text-center py-12">{{ t('history.loading') }}</div>
    <div v-else-if="error" class="text-red-400 text-center py-12">{{ error }}</div>

    <div v-else-if="stats" class="space-y-6">
      <!-- Main stat cards -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-2xl font-bold text-[var(--text-primary)]">{{ stats.total_validated ?? 0 }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.totalValidated') }}</div>
        </div>
        <div class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-2xl font-bold text-emerald-400">{{ stats.correct ?? 0 }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.correct') }}</div>
        </div>
        <div class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-2xl font-bold text-amber-400">{{ stats.partial ?? 0 }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.partial') }}</div>
        </div>
        <div class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-2xl font-bold text-red-400">{{ stats.incorrect ?? 0 }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.incorrect') }}</div>
        </div>
      </div>

      <!-- Accuracy bar -->
      <div v-if="accuracy !== null" class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)]">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-[var(--text-primary)] font-medium">{{ t('validationStats.accuracyLabel') }}</span>
          <span class="text-emerald-400 font-mono font-bold">{{ accuracy }}%</span>
        </div>
        <div class="w-full bg-[var(--border)] rounded-full h-2.5">
          <div
            class="h-2.5 rounded-full bg-emerald-500 transition-all duration-500"
            :style="{ width: `${accuracy}%` }"
          />
        </div>
      </div>

      <!-- Detection stats -->
      <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
        <div class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-xl font-bold text-blue-400">{{ stats.total_completed || 0 }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.completed') }}</div>
        </div>
        <div class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-xl font-bold text-gray-400">{{ stats.false_positives || 0 }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.fp') }}</div>
        </div>
        <div class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-xl font-bold text-red-300">{{ stats.missed_nodules || 0 }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.missed') }}</div>
        </div>
        <div class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-xl font-bold text-green-400">{{ stats.total_manual_annotations || 0 }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.manualAnn') }}</div>
        </div>
        <div class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-xl font-bold text-[var(--text-primary)]">{{ stats.total_completed || 0 }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.totalStudies') }}</div>
        </div>
        <div v-if="stats.pending_validation != null" class="bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)] text-center">
          <div class="text-xl font-bold text-yellow-400">{{ stats.pending_validation }}</div>
          <div class="text-xs text-[var(--text-secondary)] mt-1">{{ t('validationStats.pending') }}</div>
        </div>
      </div>

      <!-- Empty state -->
      <div v-if="stats.total_validated === 0" class="text-[var(--text-secondary)] text-center py-8 text-sm">
        {{ t('validationStats.noValidations') }}
      </div>
    </div>
  </div>
</template>
