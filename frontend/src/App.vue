<script setup>
import { useRouter, useRoute } from 'vue-router'
import { useTheme } from './composables/useTheme.js'
import { useConfirm } from './composables/useConfirm.js'
import { useI18n } from './i18n/index.js'
import ConfirmModal from './components/ConfirmModal.vue'

const router = useRouter()
const route = useRoute()
const { theme, toggle: toggleTheme } = useTheme()
const { show: confirmShow, config: confirmConfig, onConfirm, onCancel } = useConfirm()
const { locale, setLocale, t } = useI18n()
</script>

<template>
  <div class="min-h-screen flex flex-col">
    <!-- Header -->
    <header class="bg-[var(--bg-secondary)] border-b border-[var(--border)] px-6 py-3 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <img src="/logo-husl.png" alt="HUSL" class="h-9 rounded bg-white px-2 py-0.5" />
        <h1 class="text-lg font-semibold text-[var(--text-primary)]">{{ t('app.title') }}</h1>
        <span class="text-xs bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full font-mono">{{ t('app.version') }}</span>
      </div>
      <div class="flex items-center gap-2">
        <nav class="flex gap-1">
          <button
            @click="router.push('/')"
            :class="route.path === '/' ? 'bg-[var(--bg-card)] text-[var(--text-primary)]' : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'"
            class="px-4 py-1.5 rounded-lg text-sm font-medium transition-colors"
          >{{ t('nav.analyze') }}</button>
          <button
            @click="router.push('/history')"
            :class="route.path === '/history' ? 'bg-[var(--bg-card)] text-[var(--text-primary)]' : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'"
            class="px-4 py-1.5 rounded-lg text-sm font-medium transition-colors"
          >{{ t('nav.history') }}</button>
          <button
            @click="router.push('/validation')"
            :class="route.path === '/validation' ? 'bg-[var(--bg-card)] text-[var(--text-primary)]' : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'"
            class="px-4 py-1.5 rounded-lg text-sm font-medium transition-colors"
          >{{ t('nav.validation') }}</button>
        </nav>

        <!-- Locale selector -->
        <select
          :value="locale"
          @change="setLocale(($event.target).value)"
          class="bg-[var(--bg-card)] text-xs text-[var(--text-secondary)] border border-[var(--border)] rounded px-1.5 py-1 cursor-pointer focus:outline-none"
        >
          <option value="ca">CA</option>
          <option value="es">ES</option>
        </select>

        <!-- Theme toggle -->
        <button @click="toggleTheme" aria-label="Toggle theme" class="p-1.5 rounded-lg hover:bg-[var(--bg-card)] text-[var(--text-secondary)] transition-colors">
          <!-- Sun (show in dark mode) -->
          <svg v-if="theme === 'dark'" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
          </svg>
          <!-- Moon (show in light mode) -->
          <svg v-else class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
          </svg>
        </button>
      </div>
    </header>

    <!-- Main -->
    <main class="flex-1">
      <router-view />
    </main>

    <!-- Footer -->
    <footer class="text-center text-[10px] text-[var(--text-secondary)] py-1 border-t border-[var(--border)]">
      {{ t('footer') }}
    </footer>

    <!-- Global confirm modal -->
    <ConfirmModal
      :show="confirmShow"
      :title="confirmConfig.title"
      :message="confirmConfig.message"
      :confirm-text="confirmConfig.confirmText"
      :cancel-text="confirmConfig.cancelText"
      :variant="confirmConfig.variant"
      @confirm="onConfirm"
      @cancel="onCancel"
    />
  </div>
</template>
