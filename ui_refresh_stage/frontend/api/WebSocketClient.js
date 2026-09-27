/**
 * WebSocket 客户端 - 管理 WebSocket 连接和消息处理
 * WS 地址从 VITE_API_URL 自动推导，无需硬编码 host
 */

class WebSocketClient {
  constructor(url = null) {
    if (url) {
      this.url = url
    } else {
      // 从 VITE_API_URL 推导 WS 地址
      // VITE_API_URL 形如:
      //   开发: http://localhost:8000/agent
      //   生产: https://www.tvosagent.cn/agent
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/agent'

      // http -> ws, https -> wss
      const wsBase = apiUrl
        .replace(/^https:\/\//, 'wss://')
        .replace(/^http:\/\//, 'ws://')
        // 去掉末尾的 /agent 路径，只保留 host
        .replace(/\/agent$/, '')

      this.url = `${wsBase}/ws/agent/`
    }

    console.log('📡 WebSocket URL:', this.url)

    this.ws = null
    this.messageHandlers = []
    this.connectionHandlers = []
    this.errorHandlers = []
    this.isConnected = false
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
  }

  /**
   * 建立连接
   */
  connect() {
    return new Promise((resolve, reject) => {
      try {
        console.log('🔗 正在连接到 WebSocket...')
        this.ws = new WebSocket(this.url)

        this.ws.onopen = () => {
          console.log('✅ WebSocket 已连接')
          this.isConnected = true
          this.reconnectAttempts = 0
          this.connectionHandlers.forEach(h => h(true))
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            this.messageHandlers.forEach(h => h(data))
          } catch (e) {
            console.error('消息解析失败:', e)
          }
        }

        this.ws.onerror = (error) => {
          console.error('❌ WebSocket 错误:', error)
          this.errorHandlers.forEach(h => h(error))
          reject(error)
        }

        this.ws.onclose = () => {
          console.log('❌ WebSocket 已断开')
          this.isConnected = false
          this.connectionHandlers.forEach(h => h(false))
          this.attemptReconnect()
        }
      } catch (e) {
        console.error('WebSocket 创建失败:', e)
        reject(e)
      }
    })
  }

  /**
   * 自动重连（指数退避）
   */
  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000)
      console.log(`⏳ ${delay}ms 后重新连接 (第 ${this.reconnectAttempts}/${this.maxReconnectAttempts} 次)`)

      setTimeout(() => {
        this.connect().catch(err => {
          console.error('重连失败:', err)
        })
      }, delay)
    } else {
      console.error('❌ 达到最大重连次数，请检查后端服务')
    }
  }

  /**
   * 发送消息
   */
  send(data) {
    if (!this.isConnected) {
      console.error('WebSocket 未连接')
      return false
    }
    try {
      this.ws.send(JSON.stringify(data))
      return true
    } catch (e) {
      console.error('消息发送失败:', e)
      return false
    }
  }

  onMessage(handler)          { this.messageHandlers.push(handler) }
  onConnectionChange(handler) { this.connectionHandlers.push(handler) }
  onError(handler)            { this.errorHandlers.push(handler) }

  disconnect() {
    if (this.ws) this.ws.close()
  }
}

export default WebSocketClient
