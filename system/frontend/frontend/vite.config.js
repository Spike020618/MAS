// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,  // 如果端口被占用则失败
    
    // 允许的主机
    allowedHosts: [
      'www.tvosagent.cn',
      'tvosagent.cn',
      'localhost'
    ],
    
    // 代理配置 - 解决开发环境的跨域和WebSocket连接问题
    proxy: {
      // API 代理
      '/agent': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      // WebSocket 代理
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
        secure: false,
      }
    }
  },
  // 构建配置
  build: {
    sourcemap: true
  }
})