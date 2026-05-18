<script setup>
/**
 * HistoryView — browsable list of past CXR analyses.
 * Provides search, per-study detail panel with image viewer and detection list,
 * individual and bulk delete, and threshold-based detection filtering.
 */
import { ref, onMounted, computed, watch } from 'vue'
import { getHistory, deleteStudy, deleteAllStudies, getOriginalImageUrl } from '../lib/api.js'
import { useI18n } from '../i18n/index.js'
import { useConfirm } from '../composables/useConfirm.js'


import StudyStatus from '../components/StudyStatus.vue'
import CxrViewer from '../components/CxrViewer.vue'
import DetectionList from '../components/DetectionList.vue'
import ValidationPanel from '../components/ValidationPanel.vue'

const { t } = useI18n()
const { confirm } = useConfirm()

const studies = ref([])
const loading = ref(false)
const page = ref(1)
const searchInput = ref('')
const searchQuery = ref('')
const selectedStudy = ref(null)
const threshold = ref(0.3)
const actionError = ref(null)
const editMode = ref(false)
const manualAnnotations = ref([])
const falsePositives = ref(new Set())

watch(() => selectedStudy.value?.study_uid, () => {
  editMode.value = false
  manualAnnotations.value = []
  falsePositives.value = new Set()
})

function onEditModeChanged(mode) { editMode.value = mode }
function onDrawBox(box) { manualAnnotations.value = [...manualAnnotations.value, box] }
function onToggleFp(idx) {
  const s = new Set(falsePositives.value)
  if (s.has(idx)) s.delete(idx); else s.add(idx)
  falsePositives.value = s
}
function onDeleteAnnotation(idx) { manualAnnotations.value = manualAnnotations.value.filter((_, i) => i !== idx) }

const perPage = 20
const totalLoaded = ref(0)
const hasMore = ref(true)

async function load(query) {
  loading.value = true
  page.value = 1
  try {
    studies.value = await getHistory(1, perPage, null, query || null)
    totalLoaded.value = studies.value.length
    hasMore.value = studies.value.length >= perPage
  } catch (e) {
    studies.value = []
  } finally {
    loading.value = false
  }
}

async function loadMore() {
  const nextPage = page.value + 1
  try {
    const more = await getHistory(nextPage, perPage, null, searchQuery.value || null)
    const existingIds = new Set(studies.value.map(s => s.study_uid))
    const deduplicated = more.filter(s => !existingIds.has(s.study_uid))
    studies.value = [...studies.value, ...deduplicated]
    page.value = nextPage
    totalLoaded.value = studies.value.length
    hasMore.value = more.length >= perPage
  } catch (e) {
    actionError.value = e.message || t('errors.connectionFailed')
  }
}

onMounted(() => load(''))

// Buscar solo al pulsar Enter o al hacer click en buscar
function onSearch() {
  searchQuery.value = searchInput.value
  selectedStudy.value = null
  load(searchQuery.value)
}

function onSearchClear() {
  searchInput.value = ''
  searchQuery.value = ''
  load('')
}

function selectStudy(s) {
  selectedStudy.value = selectedStudy.value?.study_uid === s.study_uid ? null : s
  threshold.value = 0.3
}

const filteredDetections = computed(() => {
  if (!selectedStudy.value?.detections) return []
  return selectedStudy.value.detections
    .filter(d => d.score >= threshold.value)
    .sort((a, b) => b.score - a.score)
})

async function onDelete(uid) {
  const ok = await confirm({
    title: t('confirm.deleteTitle'),
    message: `${t('confirm.deleteMessage')} (${uid})`,
    confirmText: t('confirm.confirm'),
    cancelText: t('confirm.cancel'),
    variant: 'danger',
  })
  if (!ok) return
  actionError.value = null
  try {
    await deleteStudy(uid)
    if (selectedStudy.value?.study_uid === uid) selectedStudy.value = null
    await load(searchQuery.value)
  } catch (e) {
    actionError.value = e.message || t('errors.deleteFailed')
  }
}

async function onDeleteAll() {
  const ok = await confirm({
    title: t('confirm.deleteAllTitle'),
    message: t('confirm.deleteAllMessage'),
    confirmText: t('confirm.confirm'),
    cancelText: t('confirm.cancel'),
    variant: 'danger',
  })
  if (!ok) return
  actionError.value = null
  try {
    await deleteAllStudies()
    selectedStudy.value = null
    await load('')
  } catch (e) {
    actionError.value = e.message || t('errors.deleteFailed')
  }
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  const pad = n => String(n).padStart(2, '0')
  return `${pad(d.getDate())}/${pad(d.getMonth()+1)}/${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
</script>

<template>
  <div class="max-w-[1600px] mx-auto px-4 py-3">
    <!-- Header -->
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-lg font-bold text-[var(--text-primary)]">{{ t('history.title') }}</h2>
      <div class="flex items-center gap-2">
        <div class="relative flex items-center">
          <svg class="w-4 h-4 text-gray-500 absolute left-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            v-model="searchInput"
            @keydown.enter="onSearch"
            type="text"
            :placeholder="t('history.search')"
            class="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg pl-9 pr-8 py-1.5 text-sm text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none focus:border-emerald-500 w-56"
          />
          <button v-if="searchInput" @click="onSearchClear" aria-label="Clear search" class="absolute right-2 text-gray-500 hover:text-gray-300">
            <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
        <button
          @click="onDeleteAll"
          class="text-xs px-3 py-1.5 rounded-lg text-[var(--text-secondary)] bg-[var(--bg-card)] border border-[var(--border)] hover:opacity-80 transition-colors"
        >{{ t('history.deleteAll') }}</button>
      </div>
    </div>

    <div v-if="actionError" class="text-red-400 text-sm bg-red-500/10 px-4 py-2 rounded-lg mb-3">{{ actionError }}</div>

    <div v-if="loading" class="text-[var(--text-secondary)] text-center py-12">{{ t('history.loading') }}</div>

    <div v-else-if="studies.length === 0" class="text-[var(--text-secondary)] text-center py-12">
      {{ searchQuery ? t('history.noResults') : t('history.empty') }}
    </div>

    <div v-else class="grid grid-cols-1 md:grid-cols-[300px_1fr] gap-3">
      <!-- Lista -->
      <div class="space-y-1.5 overflow-y-auto pr-1" style="max-height: calc(100vh - 110px)">
        <div
          v-for="s in studies" :key="s.study_uid"
          @click="selectStudy(s)"
          :class="selectedStudy?.study_uid === s.study_uid ? 'border-emerald-500 bg-emerald-500/5' : 'border-gray-800 hover:border-gray-600'"
          class="bg-[var(--bg-secondary)] border rounded-lg p-2.5 cursor-pointer transition-colors"
        >
          <div class="flex items-center justify-between">
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-1.5">
                <span class="text-[var(--text-primary)] text-xs font-mono truncate">{{ s.study_uid }}</span>
                <span v-if="s.num_detections > 0" class="bg-red-500/10 text-red-400 text-[10px] px-1.5 py-0.5 rounded-full font-bold shrink-0">
                  {{ s.num_detections }}
                </span>
                <span v-else-if="s.status === 'completed'" class="text-emerald-400 text-[10px] shrink-0">OK</span>
                <!-- Validation badge -->
                <span v-if="s.validation?.validation_result === 'correct'" class="text-emerald-400 shrink-0" title="Validado correcto">
                  <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>
                </span>
                <span v-else-if="s.validation?.validation_result === 'partial'" class="text-amber-400 shrink-0" title="Validado parcial">
                  <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>
                </span>
                <span v-else-if="s.validation?.validation_result === 'incorrect'" class="text-red-400 shrink-0" title="Validado incorrecto">
                  <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/></svg>
                </span>
                <span v-else-if="s.status === 'completed'" class="text-gray-600 shrink-0" title="Pendiente validacion">
                  <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                </span>
              </div>
              <div class="flex items-center gap-2 mt-0.5">
                <span class="text-gray-500 text-[10px]">{{ formatDate(s.created_at) }}</span>
                <span v-if="s.inference_time_ms" class="text-gray-600 text-[10px] font-mono">{{ Math.round(s.inference_time_ms) }}ms</span>
              </div>
              <p v-if="s.patient_id" class="text-gray-400 text-[10px] mt-0.5">{{ s.patient_id }}</p>
            </div>
            <button
              @click.stop="onDelete(s.study_uid)"
              :aria-label="t('confirm.deleteTitle')"
              class="text-gray-700 hover:text-red-400 p-1 shrink-0 transition-colors"
            >
              <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>
        <!-- Load more -->
        <button v-if="hasMore" @click="loadMore"
          class="w-full text-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] text-xs py-2 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-secondary)] transition-colors"
        >{{ t('history.loadMore') }} ({{ totalLoaded }} {{ t('history.shown') }})</button>
      </div>

      <!-- Detalle -->
      <div>
        <div v-if="!selectedStudy" class="text-[var(--text-secondary)] text-center py-24 text-sm">
          {{ t('history.select') }}
        </div>

        <div v-else>
          <!-- Info + status -->
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-3">
              <h3 class="text-[var(--text-primary)] font-mono text-sm font-semibold">{{ selectedStudy.study_uid }}</h3>
              <span class="text-gray-500 text-xs">{{ formatDate(selectedStudy.created_at) }}</span>
              <span v-if="selectedStudy.patient_id" class="text-gray-400 text-xs">{{ selectedStudy.patient_id }}</span>
            </div>
            <div class="flex items-center gap-3">
              <!-- Slider inline -->
              <div v-if="selectedStudy.status === 'completed'" class="flex items-center gap-2 bg-[var(--bg-secondary)] rounded-lg px-3 py-1 border border-[var(--border)]">
                <span class="text-gray-400 text-xs">{{ t('history.threshold') }}</span>
                <input
                  type="range" min="0.01" max="0.99" step="0.01"
                  :value="threshold"
                  @input="threshold = parseFloat($event.target.value)"
                  class="w-24 h-1.5 bg-[var(--border)] rounded-full appearance-none cursor-pointer
                    [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                    [&::-webkit-slider-thumb]:bg-emerald-400 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                />
                <span class="text-emerald-400 text-xs font-mono w-8">{{ (threshold * 100).toFixed(0) }}%</span>
                <span class="text-gray-500 text-xs">{{ filteredDetections.length }} {{ t('history.det') }}</span>
              </div>
              <StudyStatus :status="selectedStudy.status" :inference-ms="selectedStudy.inference_time_ms" />
            </div>
          </div>

          <!-- Imagen + detecciones lado a lado -->
          <div class="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-3">
            <CxrViewer
              :src="getOriginalImageUrl(selectedStudy.study_uid)"
              :detections="filteredDetections"
              :edit-mode="editMode"
              :manual-annotations="manualAnnotations"
              :false-positives="falsePositives"
              @draw-box="onDrawBox"
              @toggle-fp="onToggleFp"
              @delete-annotation="onDeleteAnnotation"
              style="max-height: calc(100vh - 180px)"
            />
            <div v-if="selectedStudy.status === 'completed'" class="space-y-3 overflow-y-auto" style="max-height: calc(100vh - 180px)">
              <!-- Resumen -->
              <div class="bg-[var(--bg-secondary)] rounded-lg p-3 border border-[var(--border)]">
                <div class="grid grid-cols-2 gap-2 text-center">
                  <div class="bg-[var(--bg-card)] rounded p-2">
                    <div class="text-lg font-bold" :class="filteredDetections.length > 0 ? 'text-red-400' : 'text-emerald-400'">{{ filteredDetections.length }}</div>
                    <div class="text-[10px] text-[var(--text-secondary)]">{{ t('analyze.detections') }}</div>
                  </div>
                  <div class="bg-[var(--bg-card)] rounded p-2">
                    <div class="text-lg font-bold text-blue-400">{{ filteredDetections.length > 0 ? (Math.max(...filteredDetections.map(d=>d.score)) * 100).toFixed(0) + '%' : '-' }}</div>
                    <div class="text-[10px] text-[var(--text-secondary)]">{{ t('analyze.maxScore') }}</div>
                  </div>
                </div>
              </div>
              <!-- Lista detecciones -->
              <div class="bg-[var(--bg-secondary)] rounded-lg p-3 border border-[var(--border)]">
                <DetectionList :detections="filteredDetections" />
              </div>
              <!-- Validacion -->
              <div v-if="selectedStudy.status === 'completed'" class="bg-[var(--bg-secondary)] rounded-lg p-3 border border-[var(--border)]">
                <ValidationPanel
                  :study="selectedStudy"
                  :image-url="getOriginalImageUrl(selectedStudy.study_uid)"
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
      </div>
    </div>
  </div>
</template>
