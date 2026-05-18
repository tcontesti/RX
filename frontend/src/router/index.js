import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'analyze', component: () => import('../views/AnalyzeView.vue') },
  { path: '/history', name: 'history', component: () => import('../views/HistoryView.vue') },
  { path: '/validation', name: 'validation', component: () => import('../views/ValidationStatsView.vue') },
]

export default createRouter({ history: createWebHistory(), routes })
