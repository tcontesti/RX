<script setup>
/**
 * DropZone — drag-and-drop (or click-to-browse) file input for CXR images.
 * Emits `file-selected` with the chosen File object.
 */
import { ref } from 'vue'
import { useI18n } from '../i18n/index.js'

const { t } = useI18n()
const emit = defineEmits(['file-selected'])
const dragging = ref(false)
const fileInput = ref(null)

function onDrop(e) {
  dragging.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f) emit('file-selected', f)
}

function onSelect(e) {
  const f = e.target.files?.[0]
  if (f) emit('file-selected', f)
}
</script>

<template>
  <div
    @dragover.prevent="dragging = true"
    @dragleave="dragging = false"
    @drop.prevent="onDrop"
    @click="fileInput?.click()"
    :class="dragging ? 'border-emerald-400 bg-emerald-500/5' : 'border-[var(--border)] hover:border-[var(--text-secondary)]'"
    class="border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200"
  >
    <input ref="fileInput" type="file" accept=".png,.jpg,.jpeg,.dcm,.mha" class="hidden" @change="onSelect" />

    <div class="flex flex-col items-center gap-4">
      <div class="w-16 h-16 rounded-2xl bg-[var(--bg-card)] flex items-center justify-center">
        <svg class="w-8 h-8 text-[var(--text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      </div>
      <div>
        <p class="text-[var(--text-primary)] font-medium">{{ t('analyze.dropzone') }}</p>
        <p class="text-[var(--text-secondary)] text-sm mt-1">{{ t('analyze.dropzoneHint') }}</p>
      </div>
    </div>
  </div>
</template>
