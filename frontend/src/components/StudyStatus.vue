<script setup>
/**
 * StudyStatus — pill badge showing the current study lifecycle state
 * (idle, uploading, processing, completed, error) with optional inference time.
 */
import { useI18n } from '../i18n/index.js'

const { t } = useI18n()
const props = defineProps({ status: String, inferenceMs: Number })

const colors = {
  idle:       'bg-gray-500/10 text-gray-400',
  uploading:  'bg-blue-500/10 text-blue-400',
  processing: 'bg-amber-500/10 text-amber-400',
  completed:  'bg-emerald-500/10 text-emerald-400',
  error:      'bg-red-500/10 text-red-400',
}
</script>

<template>
  <div class="flex items-center gap-2">
    <span :class="colors[status]" class="px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-1.5">
      <span v-if="status === 'processing' || status === 'uploading'" class="w-2 h-2 bg-current rounded-full animate-pulse" />
      <span v-else-if="status === 'completed'" class="w-2 h-2 bg-current rounded-full" />
      {{ t('status.' + status) }}
    </span>
    <span v-if="inferenceMs" class="text-xs text-[var(--text-secondary)] font-mono">{{ Math.round(inferenceMs) }}ms</span>
  </div>
</template>
