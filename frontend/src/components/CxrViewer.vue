<script setup>
/**
 * CxrViewer — interactive chest X-ray image viewer with bounding-box overlay.
 * Supports pan (drag), zoom (scroll wheel), fit-to-container, and 1:1 pixel view.
 * In edit mode: zoom/pan disabled, allows drawing new boxes and marking FPs.
 * Detection boxes are color-coded by confidence (red > amber > cyan).
 */
import { ref, computed, watch } from 'vue'
import { useI18n } from '../i18n/index.js'

const { t } = useI18n()

const props = defineProps({
  src: String,
  detections: { type: Array, default: () => [] },
  imageSize: { type: Number, default: 1024 },
  editMode: { type: Boolean, default: false },
  manualAnnotations: { type: Array, default: () => [] },
  falsePositives: { type: Set, default: () => new Set() },
})

const emit = defineEmits(['draw-box', 'toggle-fp', 'delete-annotation'])

const containerEl = ref(null)
const imgEl = ref(null)
const svgEditEl = ref(null)
const scale = ref(1)
const offset = ref({ x: 0, y: 0 })
const dragging = ref(false)
const dragStart = ref({ x: 0, y: 0 })
const imgLoaded = ref(false)
const naturalW = ref(1024)
const naturalH = ref(1024)

// Drawing state (edit mode only)
const drawing = ref(false)
const drawStartPt = ref({ x: 0, y: 0 })
const drawCurrentPt = ref({ x: 0, y: 0 })

const drawRect = computed(() => {
  if (!drawing.value) return null
  const x1 = Math.min(drawStartPt.value.x, drawCurrentPt.value.x)
  const y1 = Math.min(drawStartPt.value.y, drawCurrentPt.value.y)
  const x2 = Math.max(drawStartPt.value.x, drawCurrentPt.value.x)
  const y2 = Math.max(drawStartPt.value.y, drawCurrentPt.value.y)
  return { x1, y1, x2, y2 }
})

// Reset zoom/pan when entering edit mode
watch(() => props.editMode, (editing) => {
  if (editing) {
    scale.value = 1
    offset.value = { x: 0, y: 0 }
    dragging.value = false
    drawing.value = false
  }
})

const realZoomPercent = computed(() => {
  if (!containerEl.value || !imgLoaded.value) return Math.round(scale.value * 100)
  const containerSize = Math.min(containerEl.value.clientWidth, containerEl.value.clientHeight)
  const fitScale = containerSize / Math.max(naturalW.value, naturalH.value)
  return Math.round(scale.value * fitScale * 100)
})

function onImgLoad() {
  if (imgEl.value) {
    naturalW.value = imgEl.value.naturalWidth || 1024
    naturalH.value = imgEl.value.naturalHeight || 1024
  }
  imgLoaded.value = true
}

function mouseToImageCoords(e) {
  // Try SVG matrix transform first
  const svg = svgEditEl.value
  if (svg) {
    try {
      const pt = svg.createSVGPoint()
      pt.x = e.clientX
      pt.y = e.clientY
      const ctm = svg.getScreenCTM()
      if (ctm) {
        const svgPt = pt.matrixTransform(ctm.inverse())
        return {
          x: Math.round(Math.max(0, Math.min(props.imageSize, svgPt.x))),
          y: Math.round(Math.max(0, Math.min(props.imageSize, svgPt.y))),
        }
      }
    } catch {}
  }
  // Fallback: use container bounding rect
  const container = containerEl.value
  if (!container) return null
  const rect = container.getBoundingClientRect()
  const relX = e.clientX - rect.left
  const relY = e.clientY - rect.top
  return {
    x: Math.round(Math.max(0, Math.min(props.imageSize, (relX / rect.width) * props.imageSize))),
    y: Math.round(Math.max(0, Math.min(props.imageSize, (relY / rect.height) * props.imageSize))),
  }
}

// --- View mode handlers ---
function onWheel(e) {
  if (props.editMode) return
  e.preventDefault()
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  scale.value = Math.max(0.3, Math.min(8, scale.value * delta))
}

function onMouseDown(e) {
  if (props.editMode) {
    // Start drawing
    const pt = mouseToImageCoords(e)
    if (!pt) return
    e.preventDefault()
    drawing.value = true
    drawStartPt.value = pt
    drawCurrentPt.value = pt
  } else {
    // Start panning
    dragging.value = true
    dragStart.value = { x: e.clientX - offset.value.x, y: e.clientY - offset.value.y }
  }
}

function onMouseMove(e) {
  if (props.editMode) {
    if (!drawing.value) return
    const pt = mouseToImageCoords(e)
    if (pt) drawCurrentPt.value = pt
  } else {
    if (!dragging.value) return
    offset.value = { x: e.clientX - dragStart.value.x, y: e.clientY - dragStart.value.y }
  }
}

function onMouseUp() {
  if (props.editMode) {
    if (!drawing.value) return
    // Calcular rect ANTES de poner drawing=false (drawRect depende de drawing)
    const x1 = Math.min(drawStartPt.value.x, drawCurrentPt.value.x)
    const y1 = Math.min(drawStartPt.value.y, drawCurrentPt.value.y)
    const x2 = Math.max(drawStartPt.value.x, drawCurrentPt.value.x)
    const y2 = Math.max(drawStartPt.value.y, drawCurrentPt.value.y)
    drawing.value = false
    if (x2 - x1 < 10 || y2 - y1 < 10) return
    emit('draw-box', { x1, y1, x2, y2, label: 'nodule', annotation_type: 'missed' })
  } else {
    dragging.value = false
  }
}

function onAiBoxClick(idx, e) {
  if (!props.editMode) return
  e.stopPropagation()
  emit('toggle-fp', idx)
}

function onDeleteAnnotation(idx, e) {
  e.stopPropagation()
  emit('delete-annotation', idx)
}

function resetView() {
  if (containerEl.value && imgLoaded.value) {
    const containerSize = Math.min(containerEl.value.clientWidth, containerEl.value.clientHeight)
    const fitScale = containerSize / Math.max(naturalW.value, naturalH.value)
    scale.value = 1 / fitScale
  } else {
    scale.value = 1
  }
  offset.value = { x: 0, y: 0 }
}

function fitView() {
  scale.value = 1
  offset.value = { x: 0, y: 0 }
}

function boxColor(score) {
  if (score >= 0.8) return '#ff2020'
  if (score >= 0.5) return '#ffaa00'
  return '#00ddff'
}
</script>

<template>
  <div
    ref="containerEl"
    class="relative overflow-hidden rounded-xl bg-black select-none"
    :class="editMode ? 'cursor-crosshair' : 'cursor-grab active:cursor-grabbing'"
    style="aspect-ratio: 1"
    @wheel="onWheel"
    @mousedown="onMouseDown"
    @mousemove="onMouseMove"
    @mouseup="onMouseUp"
    @mouseleave="onMouseUp"
  >
    <!-- Image -->
    <img
      v-if="src"
      ref="imgEl"
      :src="src"
      class="absolute inset-0 w-full h-full object-contain"
      :style="editMode ? {} : { transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }"
      draggable="false"
      @load="onImgLoad"
    />

    <!-- SVG Overlay: view mode (detections + manual annotations, no interaction) -->
    <svg
      v-if="!editMode && (detections.length > 0 || manualAnnotations.length > 0) && imgLoaded"
      class="absolute inset-0 w-full h-full pointer-events-none"
      :style="{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }"
      :viewBox="`0 0 ${imageSize} ${imageSize}`"
      preserveAspectRatio="xMidYMid meet"
    >
      <!-- AI detections -->
      <g v-for="(d, i) in detections" :key="'ai-view-'+i">
        <rect
          :x="d.x1" :y="d.y1"
          :width="d.x2 - d.x1" :height="d.y2 - d.y1"
          :stroke="boxColor(d.score)" stroke-width="3" fill="none" opacity="0.9"
        />
        <rect
          :x="d.x1" :y="d.y1 - 22"
          :width="Math.max(80, (d.score * 100).toFixed(0).length * 10 + 70)"
          height="22" :fill="boxColor(d.score)" opacity="0.85" rx="2"
        />
        <text :x="d.x1 + 4" :y="d.y1 - 6" fill="white" font-size="14" font-family="monospace" font-weight="bold">
          {{ t('analyze.nodule') }} {{ (d.score * 100).toFixed(0) }}%
        </text>
      </g>
      <!-- Manual annotations (green, visible in view mode too) -->
      <g v-for="(a, i) in manualAnnotations" :key="'man-view-'+i">
        <rect
          :x="a.x1" :y="a.y1"
          :width="a.x2 - a.x1" :height="a.y2 - a.y1"
          stroke="#00ff88" stroke-width="3" fill="rgba(0,255,136,0.05)" opacity="0.9"
        />
        <rect :x="a.x1" :y="a.y1 - 22" :width="a.notes ? Math.min(200, a.notes.length * 7 + 20) : 65" height="22" fill="#00ff88" opacity="0.85" rx="2" />
        <text :x="a.x1 + 4" :y="a.y1 - 6" fill="#000" font-size="13" font-family="monospace" font-weight="bold">
          {{ a.notes || t('viewer.manual') }}
        </text>
      </g>
    </svg>

    <!-- SVG Overlay: edit mode (interactive) -->
    <svg
      v-if="editMode && imgLoaded"
      ref="svgEditEl"
      class="absolute inset-0 w-full h-full"
      :viewBox="`0 0 ${imageSize} ${imageSize}`"
      preserveAspectRatio="xMidYMid meet"
    >
      <!-- AI detection boxes (clickable for FP toggle) -->
      <g v-for="(d, i) in detections" :key="'ai-'+i"
        @click="onAiBoxClick(i, $event)"
        style="cursor: pointer"
      >
        <rect
          :x="d.x1" :y="d.y1"
          :width="d.x2 - d.x1" :height="d.y2 - d.y1"
          :stroke="falsePositives.has(i) ? '#666' : boxColor(d.score)"
          :stroke-width="falsePositives.has(i) ? 2 : 3"
          :stroke-dasharray="falsePositives.has(i) ? '8,4' : 'none'"
          fill="none"
          :opacity="falsePositives.has(i) ? 0.4 : 0.9"
        />
        <!-- FP strikethrough diagonal -->
        <line v-if="falsePositives.has(i)"
          :x1="d.x1" :y1="d.y1" :x2="d.x2" :y2="d.y2"
          stroke="#666" stroke-width="2" opacity="0.5"
        />
        <!-- Label -->
        <rect
          :x="d.x1" :y="d.y1 - 22"
          :width="falsePositives.has(i) ? 50 : Math.max(80, (d.score * 100).toFixed(0).length * 10 + 70)"
          height="22"
          :fill="falsePositives.has(i) ? '#666' : boxColor(d.score)"
          :opacity="falsePositives.has(i) ? 0.5 : 0.85"
          rx="2"
        />
        <text :x="d.x1 + 4" :y="d.y1 - 6" fill="white" font-size="14" font-family="monospace" font-weight="bold">
          {{ falsePositives.has(i) ? 'FP' : `${(d.score * 100).toFixed(0)}%` }}
        </text>
      </g>

      <!-- Manual annotation boxes (green) -->
      <g v-for="(a, i) in manualAnnotations" :key="'man-'+i">
        <rect
          :x="a.x1" :y="a.y1"
          :width="a.x2 - a.x1" :height="a.y2 - a.y1"
          stroke="#00ff88" stroke-width="3"
          fill="rgba(0,255,136,0.05)" opacity="0.9"
        />
        <rect :x="a.x1" :y="a.y1 - 24" width="95" height="24" fill="#00ff88" opacity="0.85" rx="2" />
        <text :x="a.x1 + 4" :y="a.y1 - 7" fill="#000" font-size="13" font-family="monospace" font-weight="bold">{{ a.notes || t('viewer.manual') }}</text>
        <!-- Delete button -->
        <g @click="onDeleteAnnotation(i, $event)" style="cursor: pointer">
          <rect :x="a.x1 + 70" :y="a.y1 - 22" width="18" height="18" fill="#000" opacity="0.5" rx="2" />
          <text :x="a.x1 + 74" :y="a.y1 - 8" fill="white" font-size="14" font-family="monospace" font-weight="bold">x</text>
        </g>
      </g>

      <!-- Drawing rect in progress -->
      <rect v-if="drawing && drawRect"
        :x="drawRect.x1" :y="drawRect.y1"
        :width="drawRect.x2 - drawRect.x1" :height="drawRect.y2 - drawRect.y1"
        stroke="#00ff88" stroke-width="2" stroke-dasharray="8 4"
        fill="rgba(0,255,136,0.08)"
      />
    </svg>

    <!-- Edit mode banner -->
    <div v-if="editMode" class="absolute top-0 left-0 right-0 bg-green-900/80 backdrop-blur-sm px-3 py-1.5 z-20 pointer-events-none">
      <span class="text-green-300 text-xs font-medium">
        {{ t('editBanner') }}
      </span>
    </div>

    <!-- Edit mode badges -->
    <div v-if="editMode" class="absolute bottom-3 left-3 flex gap-1 z-10 pointer-events-none">
      <span v-if="detections.length > 0" class="bg-red-500/80 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">
        IA: {{ detections.length - falsePositives.size }}
      </span>
      <span v-if="manualAnnotations.length > 0" class="bg-green-500/80 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">
        Manual: {{ manualAnnotations.length }}
      </span>
      <span v-if="falsePositives.size > 0" class="bg-gray-500/80 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">
        FP: {{ falsePositives.size }}
      </span>
    </div>

    <!-- View mode controls -->
    <template v-if="!editMode">
      <div class="absolute bottom-3 right-3 flex gap-2 z-10" @mousedown.stop>
        <button @click="scale = Math.min(5, scale * 1.3)" :aria-label="t('viewer.zoomIn')" class="bg-black/60 hover:bg-black/80 rounded-lg p-1.5 text-gray-300">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" /></svg>
        </button>
        <button @click="scale = Math.max(0.5, scale * 0.7)" :aria-label="t('viewer.zoomOut')" class="bg-black/60 hover:bg-black/80 rounded-lg p-1.5 text-gray-300">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18 12H6" /></svg>
        </button>
        <button @click="fitView" :aria-label="t('viewer.fit')" class="bg-black/60 hover:bg-black/80 rounded-lg p-1.5 text-gray-300 text-xs font-mono">{{ t('viewer.fit') }}</button>
        <button @click="resetView" :aria-label="t('viewer.realSize')" class="bg-black/60 hover:bg-black/80 rounded-lg p-1.5 text-gray-300 text-xs font-mono">1:1</button>
      </div>

      <div class="absolute top-3 right-3 bg-black/60 text-gray-300 text-xs font-mono px-2 py-1 rounded z-10">
        {{ realZoomPercent }}%
      </div>

      <div v-if="detections.length > 0" class="absolute top-3 left-3 bg-red-500/80 text-white text-xs font-bold px-2 py-1 rounded z-10">
        {{ detections.length }} {{ t('analyze.nodule').toLowerCase() }}{{ detections.length > 1 ? 's' : '' }}
      </div>
    </template>
  </div>
</template>
