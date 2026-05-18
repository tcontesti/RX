<script setup>
/**
 * ConfirmModal — animated confirmation dialog replacing native window.confirm().
 * Supports danger/warning/info variants with matching colors and icons.
 * Closes on Escape, backdrop click, or button press.
 */
import { watch, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  show: Boolean,
  title: String,
  message: String,
  confirmText: { type: String, default: 'Confirmar' },
  cancelText: { type: String, default: 'Cancelar' },
  variant: { type: String, default: 'danger' },
})

const emit = defineEmits(['confirm', 'cancel'])

function onKeydown(e) {
  if (e.key === 'Escape' && props.show) emit('cancel')
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))

const variantStyles = {
  danger:  { btn: 'bg-red-600 hover:bg-red-500',    icon: 'text-red-400',   bg: 'bg-red-500/10' },
  warning: { btn: 'bg-amber-600 hover:bg-amber-500', icon: 'text-amber-400', bg: 'bg-amber-500/10' },
  info:    { btn: 'bg-blue-600 hover:bg-blue-500',   icon: 'text-blue-400',  bg: 'bg-blue-500/10' },
}
</script>

<template>
  <Teleport to="body">
    <Transition name="modal">
      <div v-if="show" class="fixed inset-0 z-50 flex items-center justify-center p-4" @click.self="emit('cancel')">
        <!-- Backdrop -->
        <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" />

        <!-- Card -->
        <div class="relative bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl shadow-2xl max-w-sm w-full p-6 space-y-4">
          <!-- Icon -->
          <div class="flex items-center gap-3">
            <div :class="[variantStyles[variant]?.bg]" class="w-10 h-10 rounded-xl flex items-center justify-center shrink-0">
              <!-- Danger: trash -->
              <svg v-if="variant === 'danger'" :class="variantStyles[variant]?.icon" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              <!-- Warning: exclamation -->
              <svg v-else-if="variant === 'warning'" :class="variantStyles[variant]?.icon" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              <!-- Info: info circle -->
              <svg v-else :class="variantStyles[variant]?.icon" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h3 class="text-[var(--text-primary)] font-semibold text-sm">{{ title }}</h3>
            </div>
          </div>

          <p class="text-[var(--text-secondary)] text-sm leading-relaxed">{{ message }}</p>

          <!-- Buttons -->
          <div class="flex gap-2 pt-1">
            <button
              @click="emit('cancel')"
              class="flex-1 py-2 rounded-lg text-sm font-medium bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] border border-[var(--border)] transition-colors"
            >{{ cancelText }}</button>
            <button
              @click="emit('confirm')"
              :class="variantStyles[variant]?.btn"
              class="flex-1 py-2 rounded-lg text-sm font-semibold text-white transition-colors"
            >{{ confirmText }}</button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.modal-enter-active { transition: all 0.2s ease-out; }
.modal-leave-active { transition: all 0.15s ease-in; }
.modal-enter-from { opacity: 0; }
.modal-enter-from > div:last-child { transform: scale(0.95); }
.modal-leave-to { opacity: 0; }
.modal-leave-to > div:last-child { transform: scale(0.95); }
</style>
