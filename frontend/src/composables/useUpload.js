/**
 * Upload composable — manages CXR image upload lifecycle.
 * Handles file selection, upload to backend, polling for results,
 * and cleanup on unmount.
 *
 * @module useUpload
 */
import { ref, onUnmounted } from 'vue'
import { uploadCxr, getResults } from '../lib/api.js'
import { useI18n } from '../i18n/index.js'

/** @constant {number} Maximum allowed file size in MB. */
const MAX_FILE_SIZE_MB = 50
/** @constant {number} Consecutive poll errors before giving up. */
const MAX_POLL_ERRORS = 10
/** @constant {number} Interval between result polling requests in ms. */
const POLL_INTERVAL_MS = 1500

/**
 * Composable that encapsulates the full upload-and-poll workflow.
 *
 * @returns {{
 *   file: import('vue').Ref<File|null>,
 *   preview: import('vue').Ref<string|null>,
 *   studyUid: import('vue').Ref<string|null>,
 *   result: import('vue').Ref<Object|null>,
 *   status: import('vue').Ref<'idle'|'uploading'|'processing'|'completed'|'error'>,
 *   error: import('vue').Ref<string|null>,
 *   setFile: (f: File) => void,
 *   submit: (patientId?: string, patientName?: string) => Promise<void>,
 *   reset: () => void
 * }}
 */
export function useUpload() {
  const { t } = useI18n()
  const file = ref(null)
  const preview = ref(null)
  const studyUid = ref(null)
  const result = ref(null)
  const status = ref('idle')
  const error = ref(null)
  const pollTimer = ref(null)
  let pollErrors = 0

  /**
   * Set the selected file, validate size, and create an object URL preview.
   * Revokes any previous preview URL to prevent memory leaks.
   *
   * @param {File} f - The image file selected by the user.
   */
  function setFile(f) {
    // Revocar URL anterior para evitar memory leak
    if (preview.value) URL.revokeObjectURL(preview.value)

    // Validar tamaño
    if (f.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      error.value = t('errors.fileTooLarge').replace('{max}', MAX_FILE_SIZE_MB)
      return
    }

    file.value = f
    preview.value = URL.createObjectURL(f)
    result.value = null
    studyUid.value = null
    status.value = 'idle'
    error.value = null
  }

  /**
   * Upload the current file to the backend and begin polling for results.
   *
   * @param {string|null} patientId - Optional patient identifier.
   * @param {string|null} patientName - Optional patient name.
   */
  async function submit(patientId, patientName) {
    if (!file.value || status.value !== 'idle') return
    status.value = 'uploading'
    error.value = null
    pollErrors = 0
    try {
      const res = await uploadCxr(file.value, patientId, patientName)
      studyUid.value = res.study_uid
      status.value = 'processing'
      startPolling()
    } catch (e) {
      status.value = 'error'
      error.value = e.message || t('errors.uploadFailed')
    }
  }

  /** Start interval-based polling for study results. Stops on completion, error, or max retries. */
  function startPolling() {
    stopPolling()
    pollTimer.value = setInterval(async () => {
      try {
        const res = await getResults(studyUid.value)
        pollErrors = 0
        if (res.status === 'completed') {
          result.value = res
          status.value = 'completed'
          stopPolling()
        } else if (res.status === 'error') {
          error.value = res.error_message || t('errors.analysisFailed')
          status.value = 'error'
          stopPolling()
        }
      } catch (e) {
        pollErrors++
        if (pollErrors >= MAX_POLL_ERRORS) {
          error.value = t('errors.connectionFailed')
          status.value = 'error'
          stopPolling()
        }
      }
    }, POLL_INTERVAL_MS)
  }

  /** Clear the polling interval if active. */
  function stopPolling() {
    if (pollTimer.value) {
      clearInterval(pollTimer.value)
      pollTimer.value = null
    }
  }

  /** Reset all state to initial values, stop polling, and revoke the preview URL. */
  function reset() {
    stopPolling()
    if (preview.value) URL.revokeObjectURL(preview.value)
    file.value = null
    preview.value = null
    studyUid.value = null
    result.value = null
    status.value = 'idle'
    error.value = null
    pollErrors = 0
  }

  // Limpiar polling al desmontar componente
  onUnmounted(() => {
    stopPolling()
    if (preview.value) URL.revokeObjectURL(preview.value)
  })

  return { file, preview, studyUid, result, status, error, setFile, submit, reset }
}
