/**
 * CXR API client — thin wrapper around fetch for the chest X-ray backend.
 * All functions target the /api/cxr REST endpoints.
 * @module api
 */

const BASE = '/api/cxr'

/**
 * Parse a fetch Response, throwing on non-OK status.
 * Attempts to extract a `detail` field from the JSON error body.
 *
 * @param {Response} res - Fetch Response object.
 * @returns {Promise<Object>} Parsed JSON body.
 * @throws {Error} With the server-provided detail or statusText.
 */
async function handleResponse(res) {
  if (!res.ok) {
    let msg = res.statusText
    try { msg = (await res.json()).detail || msg } catch {}
    throw new Error(msg)
  }
  return res.json()
}

/**
 * Upload a chest X-ray image for analysis.
 *
 * @param {File} file - Image file (PNG, JPG, DICOM, MHA).
 * @param {string|null} patientId - Optional patient identifier.
 * @param {string|null} patientName - Optional patient name.
 * @returns {Promise<{study_uid: string}>} Created study metadata.
 */
export async function uploadCxr(file, patientId, patientName) {
  const fd = new FormData()
  fd.append('file', file)
  if (patientId) fd.append('patient_id', patientId)
  if (patientName) fd.append('patient_name', patientName)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: fd })
  return handleResponse(res)
}

/**
 * Fetch analysis results for a study.
 *
 * @param {string} studyUid - Unique study identifier.
 * @returns {Promise<Object>} Study results including status, detections, and timing.
 */
export async function getResults(studyUid) {
  const res = await fetch(`${BASE}/results/${studyUid}`)
  return handleResponse(res)
}

/**
 * Build the URL for the original (unprocessed) CXR image.
 *
 * @param {string} studyUid - Unique study identifier.
 * @returns {string} Absolute URL path to the original image.
 */
export function getOriginalImageUrl(studyUid) {
  return `${BASE}/results/${studyUid}/original`
}

/**
 * Retrieve paginated study history with optional filters.
 *
 * @param {number} [page=1] - Page number (1-based).
 * @param {number} [perPage=20] - Results per page.
 * @param {string|null} [status=null] - Filter by study status.
 * @param {string|null} [search=null] - Free-text search query.
 * @returns {Promise<Array<Object>>} Array of study summary objects.
 */
export async function getHistory(page = 1, perPage = 20, status = null, search = null) {
  const params = new URLSearchParams({ page, per_page: perPage })
  if (status) params.set('status', status)
  if (search) params.set('search', search)
  const res = await fetch(`${BASE}/history?${params}`)
  return handleResponse(res)
}

/**
 * Fetch aggregate statistics (total studies, completed, with nodules, avg time).
 *
 * @returns {Promise<Object>} Stats object with totals and averages.
 */
export async function getStats() {
  const res = await fetch(`${BASE}/stats`)
  return handleResponse(res)
}

/**
 * Delete a single study by its UID.
 *
 * @param {string} studyUid - Unique study identifier to delete.
 * @returns {Promise<Object>} Deletion confirmation.
 */
export async function deleteStudy(studyUid) {
  const res = await fetch(`${BASE}/results/${studyUid}`, { method: 'DELETE' })
  return handleResponse(res)
}

/**
 * Delete all studies from the system.
 *
 * @returns {Promise<Object>} Deletion confirmation.
 */
export async function deleteAllStudies() {
  const res = await fetch(`${BASE}/all`, { method: 'DELETE' })
  return handleResponse(res)
}

/**
 * Submit radiologist validation for a study.
 *
 * @param {string} studyUid - Unique study identifier.
 * @param {Object} validation - Validation payload.
 * @param {string} validation.validation_result - "correct", "partial", or "incorrect".
 * @param {string} [validation.validated_by] - Radiologist name/ID.
 * @param {string} [validation.notes] - Free-text notes.
 * @param {Array<Object>} [validation.annotations] - Manual bounding-box annotations.
 * @returns {Promise<Object>} Saved validation record.
 */
export async function validateStudy(studyUid, validation) {
  const res = await fetch(`${BASE}/results/${studyUid}/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(validation),
  })
  return handleResponse(res)
}

/**
 * Get existing validation for a study. Returns null if not yet validated.
 *
 * @param {string} studyUid - Unique study identifier.
 * @returns {Promise<Object|null>} Validation record or null.
 */
export async function getValidation(studyUid) {
  const res = await fetch(`${BASE}/results/${studyUid}/validation`)
  if (res.status === 404) return null
  return handleResponse(res)
}

/**
 * Get aggregate validation statistics.
 *
 * @returns {Promise<Object>} Stats including totals, accuracy, FP/FN counts.
 */
export async function getValidationStats() {
  const res = await fetch(`${BASE}/validation/stats`)
  return handleResponse(res)
}

/**
 * Build the URL for exporting the validated dataset.
 *
 * @param {string} [format='csv'] - Export format ('csv' or 'json').
 * @returns {string} URL to download the export file.
 */
export function getExportUrl(format = 'csv') {
  return `${BASE}/validation/export?format=${format}`
}
