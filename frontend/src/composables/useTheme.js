/**
 * Theme composable — toggles dark/light mode with localStorage persistence.
 * Applies 'dark' or 'light' class on the document root element.
 * @module useTheme
 */
import { ref, watchEffect } from 'vue'

const theme = ref(localStorage.getItem('cxr-theme') || 'dark')

export function useTheme() {
  function toggle() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
    localStorage.setItem('cxr-theme', theme.value)
  }

  watchEffect(() => {
    document.documentElement.classList.toggle('dark', theme.value === 'dark')
    document.documentElement.classList.toggle('light', theme.value === 'light')
  })

  return { theme, toggle }
}
