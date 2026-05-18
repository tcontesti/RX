<script setup>
/**
 * ThresholdSlider — range input for adjusting the detection confidence threshold.
 * Uses v-model via modelValue/update:modelValue. Displays current percentage and detection count.
 */
import { useI18n } from '../i18n/index.js'

const { t } = useI18n()
const props = defineProps({ modelValue: Number, count: Number })
const emit = defineEmits(['update:modelValue'])
</script>

<template>
  <div class="space-y-2">
    <div class="flex items-center justify-between text-sm">
      <span class="text-[var(--text-secondary)]">{{ t('analyze.confidence') }}</span>
      <span class="font-mono text-emerald-400 font-semibold">{{ (modelValue * 100).toFixed(0) }}%</span>
    </div>
    <input
      type="range"
      min="0.01" max="0.99" step="0.01"
      :value="modelValue"
      @input="emit('update:modelValue', parseFloat($event.target.value))"
      class="w-full h-2 bg-[var(--border)] rounded-full appearance-none cursor-pointer
             [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
             [&::-webkit-slider-thumb]:bg-emerald-400 [&::-webkit-slider-thumb]:rounded-full
             [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-lg"
    />
    <div class="flex justify-between text-xs text-[var(--text-secondary)]">
      <span>{{ t('analyze.moreDetections') }}</span>
      <span>{{ count }} {{ t('analyze.nodule').toLowerCase() }}{{ count !== 1 ? 's' : '' }} {{ t('analyze.detected') }}{{ count !== 1 ? 's' : '' }}</span>
      <span>{{ t('analyze.morePrecision') }}</span>
    </div>
  </div>
</template>
