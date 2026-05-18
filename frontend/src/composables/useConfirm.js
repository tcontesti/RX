/**
 * Confirm dialog composable — promise-based replacement for window.confirm().
 * Shared singleton state so the modal can live in App.vue while being triggered anywhere.
 * @module useConfirm
 */
import { ref } from 'vue'

const show = ref(false)
const config = ref({})
let resolvePromise = null

export function useConfirm() {
  function confirm({ title, message, confirmText, cancelText, variant }) {
    config.value = { title, message, confirmText, cancelText, variant: variant || 'danger' }
    show.value = true
    return new Promise(resolve => { resolvePromise = resolve })
  }

  function onConfirm() {
    show.value = false
    resolvePromise?.(true)
    resolvePromise = null
  }

  function onCancel() {
    show.value = false
    resolvePromise?.(false)
    resolvePromise = null
  }

  return { show, config, confirm, onConfirm, onCancel }
}
