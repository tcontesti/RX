<script setup>
/**
 * ValidationPanel — radiologist validation workflow.
 * After saving, controls are locked. "Editar" unlocks for modifications.
 */
import { ref, watch, onMounted } from 'vue'
import { validateStudy, getValidation } from '../lib/api.js'
import { useI18n } from '../i18n/index.js'

const { t } = useI18n()

const props = defineProps({
  study: Object,
  imageUrl: String,
  manualAnnotations: { type: Array, default: () => [] },
  falsePositives: { type: Set, default: () => new Set() },
})

const emit = defineEmits(['edit-mode-changed', 'validated', 'update:annotations', 'update:false-positives'])

const validationResult = ref(null)
const notes = ref('')
const validatedBy = ref('')
const saving = ref(false)
const saved = ref(false)
const locked = ref(false) // true = validado y bloqueado, necesita "Editar" para modificar
const saveError = ref(null)
const existingValidation = ref(null)
const editActive = ref(false)

async function loadExisting() {
  if (!props.study?.study_uid) return
  try {
    const v = await getValidation(props.study.study_uid)
    if (v) {
      existingValidation.value = v
      validationResult.value = v.validation_result
      notes.value = v.notes || ''
      validatedBy.value = v.validated_by || ''
      saved.value = true
      locked.value = true // Bloqueado al cargar

      const savedAnnotations = (v.manual_annotations || [])
        .filter(a => a.annotation_type !== 'false_positive')
        .map(a => ({ x1: a.x1, y1: a.y1, x2: a.x2, y2: a.y2, label: a.label, annotation_type: a.annotation_type, notes: a.notes || '' }))
      if (savedAnnotations.length > 0) {
        emit('update:annotations', savedAnnotations)
      }

      // Restore false positives by matching saved FP annotations to current detections
      const fpAnnotations = (v.manual_annotations || []).filter(a => a.annotation_type === 'false_positive')
      if (fpAnnotations.length > 0 && props.study?.detections) {
        const restoredFps = new Set()
        fpAnnotations.forEach(fp => {
          const idx = props.study.detections.findIndex(d =>
            Math.abs(d.x1 - fp.x1) < 5 && Math.abs(d.y1 - fp.y1) < 5 &&
            Math.abs(d.x2 - fp.x2) < 5 && Math.abs(d.y2 - fp.y2) < 5
          )
          if (idx >= 0) restoredFps.add(idx)
        })
        if (restoredFps.size > 0) emit('update:false-positives', restoredFps)
      }
    }
  } catch {}
}

onMounted(loadExisting)
watch(() => props.study?.study_uid, loadExisting)

function unlock() {
  locked.value = false
  saved.value = false
}

function selectResult(r) {
  if (locked.value) return
  validationResult.value = r
  saved.value = false
  if (r === 'partial' || r === 'incorrect') {
    editActive.value = true
    emit('edit-mode-changed', true)
  } else {
    editActive.value = false
    emit('edit-mode-changed', false)
  }
}

function exitEditMode() {
  editActive.value = false
  emit('edit-mode-changed', false)
}

function deleteAnnotation(idx) {
  if (locked.value) return
  const updated = props.manualAnnotations.filter((_, i) => i !== idx)
  emit('update:annotations', updated)
}

function updateAnnotationNote(idx, note) {
  if (locked.value) return
  const updated = props.manualAnnotations.map((a, i) => i === idx ? { ...a, notes: note } : a)
  emit('update:annotations', updated)
}

function undoFp(idx) {
  if (locked.value) return
  const newSet = new Set(props.falsePositives)
  newSet.delete(idx)
  emit('update:false-positives', newSet)
}

async function save() {
  if (!validationResult.value || !props.study?.study_uid) return
  saving.value = true
  saveError.value = null
  try {
    const payload = {
      validation_result: validationResult.value,
      validated_by: validatedBy.value || undefined,
      notes: notes.value || undefined,
      annotations: [
        ...props.manualAnnotations.map(a => ({
          x1: a.x1, y1: a.y1, x2: a.x2, y2: a.y2,
          label: a.label || 'nodule',
          annotation_type: a.annotation_type || 'missed',
          notes: a.notes || undefined,
        })),
        ...[...props.falsePositives].map(idx => {
          const d = props.study?.detections?.[idx]
          if (!d) return null
          return { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, label: 'nodule', annotation_type: 'false_positive' }
        }).filter(Boolean),
      ],
    }
    await validateStudy(props.study.study_uid, payload)
    saved.value = true
    locked.value = true
    editActive.value = false
    emit('edit-mode-changed', false)
    emit('validated', validationResult.value)
  } catch (e) {
    saveError.value = e.message || t('errors.saveFailed')
  } finally {
    saving.value = false
  }
}

function resultConfig(r) {
  const labels = {
    correct: t('validation.correct'),
    partial: t('validation.partial'),
    incorrect: t('validation.incorrect'),
  }
  return {
    correct:   { label: labels.correct,   color: 'bg-emerald-600 hover:bg-emerald-500', activeColor: 'bg-emerald-500 ring-2 ring-emerald-300' },
    partial:   { label: labels.partial,    color: 'bg-amber-600 hover:bg-amber-500',     activeColor: 'bg-amber-500 ring-2 ring-amber-300' },
    incorrect: { label: labels.incorrect, color: 'bg-red-600 hover:bg-red-500',         activeColor: 'bg-red-500 ring-2 ring-red-300' },
  }[r]
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  const pad = n => String(n).padStart(2, '0')
  return `${pad(d.getDate())}/${pad(d.getMonth()+1)}/${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
</script>

<template>
  <div class="space-y-3">
    <!-- Saved + locked indicator -->
    <div v-if="saved && locked" class="bg-[var(--bg-card)] rounded-lg p-2 border border-[var(--border)]">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span
            :class="{
              'bg-emerald-500/20 text-emerald-400': validationResult === 'correct',
              'bg-amber-500/20 text-amber-400': validationResult === 'partial',
              'bg-red-500/20 text-red-400': validationResult === 'incorrect',
            }"
            class="text-xs px-2 py-0.5 rounded-full font-semibold"
          >{{ resultConfig(validationResult)?.label }}</span>
          <span v-if="existingValidation?.validated_by" class="text-gray-500 text-[10px]">
            {{ t('validation.by') }} {{ existingValidation.validated_by }}
          </span>
          <span v-if="existingValidation?.validated_at" class="text-gray-600 text-[10px]">
            {{ formatDate(existingValidation.validated_at) }}
          </span>
        </div>
        <button @click="unlock" class="text-blue-400 hover:text-blue-300 text-[10px] font-medium transition-colors">
          {{ t('validation.edit') }}
        </button>
      </div>
      <!-- Show notes if any -->
      <p v-if="notes" class="text-gray-400 text-[10px] mt-1 italic">{{ notes }}</p>
      <!-- Show annotation count -->
      <div v-if="manualAnnotations.length > 0 || falsePositives.size > 0" class="flex gap-2 mt-1">
        <span v-if="manualAnnotations.length" class="text-green-400 text-[10px]">{{ manualAnnotations.length }} {{ t('validation.annotationCount') }}</span>
        <span v-if="falsePositives.size" class="text-gray-400 text-[10px]">{{ falsePositives.size }} FP</span>
      </div>
    </div>

    <!-- Editable form (only visible when NOT locked) -->
    <template v-if="!locked">
      <!-- Validation buttons -->
      <div>
        <h4 class="text-[10px] font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-2">{{ t('validation.title') }}</h4>
        <div class="flex gap-2">
          <button
            v-for="r in ['correct', 'partial', 'incorrect']" :key="r"
            @click="selectResult(r)"
            :class="validationResult === r ? resultConfig(r).activeColor : resultConfig(r).color"
            class="flex-1 text-white text-xs font-semibold py-2 rounded-lg transition-all"
          >{{ resultConfig(r).label }}</button>
        </div>
      </div>

      <!-- Edit mode badge -->
      <div v-if="editActive" class="bg-green-900/30 border border-green-800/50 rounded-lg p-2">
        <div class="flex items-center justify-between">
          <span class="text-green-400 text-xs font-medium">{{ t('validation.editMode') }}</span>
          <button @click="exitEditMode" class="text-green-500 hover:text-green-300 text-xs transition-colors">{{ t('validation.exit') }}</button>
        </div>
      </div>

      <!-- Manual annotations with per-box notes -->
      <div v-if="manualAnnotations.length > 0" class="space-y-2">
        <h4 class="text-[10px] font-semibold text-[var(--text-secondary)] uppercase tracking-wider">{{ t('validation.annotations') }} ({{ manualAnnotations.length }})</h4>
        <div v-for="(a, i) in manualAnnotations" :key="'ann-'+i"
          class="bg-[var(--bg-card)] rounded-lg p-2 border border-green-900/30 space-y-1"
        >
          <div class="flex items-center justify-between">
            <span class="text-green-400 text-[10px] font-mono">#{{ i+1 }}: ({{ a.x1 }},{{ a.y1 }})-({{ a.x2 }},{{ a.y2 }})</span>
            <button @click="deleteAnnotation(i)" :aria-label="'Delete annotation ' + (i+1)" class="text-gray-600 hover:text-red-400 transition-colors">
              <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <input
            :value="a.notes || ''"
            @input="updateAnnotationNote(i, $event.target.value)"
            type="text"
            :placeholder="t('validation.noteHint')"
            class="w-full bg-[var(--bg-secondary)] border border-[var(--border)] rounded px-2 py-1 text-[10px] text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none focus:border-green-500"
          />
        </div>
      </div>

      <!-- FP summary -->
      <div v-if="falsePositives.size > 0" class="space-y-1">
        <h4 class="text-[10px] font-semibold text-[var(--text-secondary)] uppercase tracking-wider">{{ t('validation.falsePositives') }} ({{ falsePositives.size }})</h4>
        <div v-for="idx in [...falsePositives]" :key="'fp-'+idx"
          class="flex items-center justify-between bg-[var(--bg-card)] rounded px-2 py-1 border border-[var(--border)]"
        >
          <span class="text-gray-400 text-[10px] font-mono line-through">
            IA box {{ study?.detections?.[idx]?.score ? (study.detections[idx].score * 100).toFixed(0) + '%' : '?' }}
          </span>
          <button @click="undoFp(idx)" class="text-gray-600 hover:text-amber-400 text-[10px] transition-colors">{{ t('validation.undo') }}</button>
        </div>
      </div>

      <!-- Radiologist -->
      <div>
        <label class="text-[10px] text-[var(--text-secondary)] block mb-1">{{ t('validation.radiologist') }}</label>
        <input v-model="validatedBy" type="text" :placeholder="t('validation.nameHint')"
          class="w-full bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-2 py-1.5 text-xs text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none focus:border-emerald-500" />
      </div>

      <!-- General notes -->
      <div>
        <label class="text-[10px] text-[var(--text-secondary)] block mb-1">{{ t('validation.notes') }}</label>
        <textarea v-model="notes" :placeholder="t('validation.obsHint')" rows="3"
          class="w-full bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-2 py-1.5 text-xs text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none focus:border-emerald-500 resize-none" />
      </div>

      <!-- Save -->
      <button @click="save" :disabled="!validationResult || saving"
        class="w-full py-2 rounded-lg text-sm font-semibold transition-colors"
        :class="!validationResult || saving ? 'bg-gray-700 text-gray-500 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-500 text-white'"
      >{{ saving ? t('validation.saving') : saved ? t('validation.update') : t('validation.save') }}</button>

      <div v-if="saveError" class="text-red-400 text-[10px] bg-red-500/10 rounded px-2 py-1">{{ saveError }}</div>
    </template>
  </div>
</template>
