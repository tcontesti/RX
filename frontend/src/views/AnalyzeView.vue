<script setup>
/**
 * AnalyzeView — primary analysis workflow view.
 * Orchestrates file selection (DropZone), upload + polling (useUpload),
 * threshold filtering (useDetections), result display (CxrViewer, DetectionList),
 * and radiologist validation with annotation editing on the main image.
 */
import { ref, watch } from 'vue'
import { useI18n } from '../i18n/index.js'
import DropZone from '../components/DropZone.vue'
import CxrViewer from '../components/CxrViewer.vue'
import ThresholdSlider from '../components/ThresholdSlider.vue'
import DetectionList from '../components/DetectionList.vue'
import StudyStatus from '../components/StudyStatus.vue'
import StatsBar from '../components/StatsBar.vue'
import ValidationPanel from '../components/ValidationPanel.vue'
import { useUpload } from '../composables/useUpload.js'
import { useDetections } from '../composables/useDetections.js'

const { t } = useI18n()
const { file, preview, result, status, error, setFile, submit, reset } = useUpload()
const { threshold, filtered, count, maxScore } = useDetections(result)

const statsRefreshKey = ref(0)
watch(status, (val) => { if (val === 'completed') statsRefreshKey.value++ })
const editMode = ref(false)
const manualAnnotations = ref([])
const falsePositives = ref(new Set())

function onFile(f) {
  setFile(f)
}

function analyze() {
  submit(null, null)
}

function onEditModeChanged(mode) {
  editMode.value = mode
}

function onDrawBox(box) {
  manualAnnotations.value = [...manualAnnotations.value, box]
}

function onToggleFp(idx) {
  const newSet = new Set(falsePositives.value)
  if (newSet.has(idx)) newSet.delete(idx)
  else newSet.add(idx)
  falsePositives.value = newSet
}

function onDeleteAnnotation(idx) {
  manualAnnotations.value = manualAnnotations.value.filter((_, i) => i !== idx)
}
</script>

<template>
  <div class="max-w-[1600px] mx-auto px-4 py-3">
    <!-- Stats + Actions + Status -->
    <div class="mb-3 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <StatsBar :refresh-key="statsRefreshKey" />
        <button
          v-if="file && status === 'idle'"
          @click="analyze"
          class="bg-emerald-500 hover:bg-emerald-400 text-white font-semibold px-5 py-1.5 rounded-lg transition-colors text-sm"
        >{{ t('analyze.analyze') }}</button>
        <button
          v-if="file && (status === 'completed' || status === 'error')"
          @click="reset"
          class="bg-[var(--bg-card)] hover:opacity-80 text-[var(--text-secondary)] font-medium px-5 py-1.5 rounded-lg transition-colors text-sm border border-[var(--border)]"
        >{{ t('analyze.newImage') }}</button>
        <!-- Analizando inline -->
        <div v-if="status === 'processing'" class="inline-flex items-center gap-2 text-amber-400 bg-amber-500/10 px-4 py-1.5 rounded-lg">
          <svg class="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" class="opacity-20" />
            <path d="M4 12a8 8 0 018-8" stroke="currentColor" stroke-width="3" stroke-linecap="round" />
          </svg>
          <span class="text-sm font-medium">{{ t('analyze.analyzing') }}</span>
        </div>
        <!-- Error inline -->
        <div v-if="error" class="text-red-400 text-sm bg-red-500/10 px-4 py-1.5 rounded-lg">{{ error }}</div>
      </div>
      <StudyStatus :status="status" :inference-ms="result?.inference_time_ms" />
    </div>

    <!-- Upload -->
    <div v-if="!file" class="max-w-2xl mx-auto mt-12">
      <DropZone @file-selected="onFile" />
    </div>

    <!-- Resultado -->
    <div v-else class="grid grid-cols-1 lg:grid-cols-[auto_320px] gap-3 max-w-[1200px] mx-auto">
      <!-- Imagen -->
      <div>
        <CxrViewer
          :src="preview"
          :detections="filtered"
          :edit-mode="editMode"
          :manual-annotations="manualAnnotations"
          :false-positives="falsePositives"
          @draw-box="onDrawBox"
          @toggle-fp="onToggleFp"
          @delete-annotation="onDeleteAnnotation"
          style="max-height: calc(100vh - 140px)"
        />
      </div>

      <!-- Panel derecho -->
      <div class="space-y-3 overflow-y-auto" style="max-height: calc(100vh - 120px)">
        <!-- Info archivo -->
        <div class="bg-[var(--bg-secondary)] rounded-lg p-3 border border-[var(--border)]">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-[var(--text-primary)] font-medium truncate">{{ file?.name }}</p>
              <p class="text-xs text-[var(--text-secondary)]">{{ file?.size ? (file.size / 1024).toFixed(0) + ' KB' : '' }}</p>
            </div>
            <span v-if="status === 'completed'"
              :class="count > 0 ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'"
              class="text-xs px-2 py-0.5 rounded-full font-semibold shrink-0"
            >{{ count > 0 ? `${count} ${t('analyze.nodule').toLowerCase()}${count > 1 ? 's' : ''}` : t('analyze.normal') }}</span>
          </div>
        </div>

        <!-- Threshold slider -->
        <div v-if="status === 'completed'" class="bg-[var(--bg-secondary)] rounded-lg p-3 border border-[var(--border)]">
          <ThresholdSlider v-model="threshold" :count="count" />
        </div>

        <!-- Resumen -->
        <div v-if="status === 'completed'" class="bg-[var(--bg-secondary)] rounded-lg p-3 border border-[var(--border)]">
          <div class="grid grid-cols-2 gap-2 text-center">
            <div class="bg-[var(--bg-card)] rounded-lg p-2">
              <div class="text-lg font-bold" :class="count > 0 ? 'text-red-400' : 'text-emerald-400'">{{ count }}</div>
              <div class="text-xs text-[var(--text-secondary)]">{{ t('analyze.detections') }}</div>
            </div>
            <div class="bg-[var(--bg-card)] rounded-lg p-2">
              <div class="text-lg font-bold text-blue-400">{{ maxScore > 0 ? (maxScore * 100).toFixed(0) + '%' : '-' }}</div>
              <div class="text-xs text-[var(--text-secondary)]">{{ t('analyze.maxScore') }}</div>
            </div>
          </div>
        </div>

        <!-- Detecciones -->
        <div v-if="status === 'completed'" class="bg-[var(--bg-secondary)] rounded-lg p-3 border border-[var(--border)]">
          <DetectionList :detections="filtered" />
        </div>

        <!-- Modelos -->
        <div v-if="status === 'completed' && result?.model_details" class="bg-[var(--bg-secondary)] rounded-lg p-3 border border-[var(--border)]">
          <h3 class="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-2">{{ t('analyze.models') }}</h3>
          <div class="space-y-1 text-xs">
            <div v-for="(info, name) in result.model_details" :key="name" class="flex justify-between text-[var(--text-secondary)]">
              <span class="font-mono uppercase">{{ name }}</span>
              <span>{{ info.num_detections }} det / {{ info.time_ms }}ms</span>
            </div>
            <div class="flex justify-between text-emerald-400 font-medium border-t border-gray-700 pt-1.5 mt-1.5">
              <span>Ensemble WBF</span>
              <span>{{ result.inference_time_ms }}ms</span>
            </div>
          </div>
        </div>

        <!-- Validacion -->
        <div v-if="status === 'completed'" class="bg-[var(--bg-secondary)] rounded-lg p-3 border border-[var(--border)]">
          <ValidationPanel
            :study="result"
            :image-url="preview"
            :manual-annotations="manualAnnotations"
            :false-positives="falsePositives"
            @edit-mode-changed="onEditModeChanged"
            @update:annotations="manualAnnotations = $event"
            @update:false-positives="falsePositives = $event"
          />
        </div>
      </div>
    </div>
  </div>
</template>
