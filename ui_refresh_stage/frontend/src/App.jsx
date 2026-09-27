import { useEffect, useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Header from './components/Header'
import WebSocketClient from '../api/WebSocketClient'

function App() {
  const [ws, setWs] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [apiAvailable, setApiAvailable] = useState(false)
  const [searchResults, setSearchResults] = useState(null)
  // 全局聊天记录，所有来源的对话都追加到这里
  const [chatHistory, setChatHistory] = useState([])

  useEffect(() => {
    // 推荐搜索使用 HTTP；WebSocket 是旧版可选能力。单独探测 HTTP，避免误报“已断开”。
    const apiBase = (import.meta.env.VITE_API_URL || 'http://localhost:8000/agent').replace(/\/agent\/?$/, '')
    const checkApi = () => fetch(`${apiBase}/health`, { cache: 'no-store' })
      .then((response) => setApiAvailable(response.ok))
      .catch(() => setApiAvailable(false))
    checkApi()
    const apiTimer = window.setInterval(checkApi, 10000)

    const client = new WebSocketClient()

    client.onConnectionChange((connected) => {
      setIsConnected(connected)
    })

    client.onMessage((data) => {
      console.log('📨 WebSocket 收到消息:', data)

      const timestamp = new Date().toLocaleTimeString('zh-CN')

      if (data.type === 'media_search') {
        // 影视搜索场景：更新媒体列表 + 追加对话记录
        setSearchResults(data)
        if (data.chat) {
          setChatHistory(prev => [
            ...prev,
            {
              id: Date.now(),
              role: 'user',
              content: data.query || '搜索请求',
              timestamp,
              source: data.source || 'search'
            },
            {
              id: Date.now() + 1,
              role: 'assistant',
              content: data.chat,
              timestamp,
              source: data.source || 'search',
              hasMedia: true,
              mediaCount: data.medias?.length || 0
            }
          ])
        }
      } else if (data.type === 'chat') {
        // 纯对话场景：同时显示用户问句和助手回复
        if (data.chat) {
          const msgs = []
          if (data.query) {
            msgs.push({
              id: Date.now(),
              role: 'user',
              content: data.query,
              timestamp,
              source: data.source || 'chat'
            })
          }
          msgs.push({
            id: Date.now() + 1,
            role: 'assistant',
            content: data.chat,
            timestamp,
            source: data.source || 'chat'
          })
          setChatHistory(prev => [...prev, ...msgs])
        }
      } else if (data.type === 'ack') {
        // 心跳确认，忽略
      }
    })

    client.onError((error) => {
      console.error('❌ WebSocket 错误:', error)
    })

    client.connect().catch((err) => {
      console.error('连接失败:', err)
    })

    setWs(client)

    return () => {
      if (client) client.disconnect()
      window.clearInterval(apiTimer)
    }
  }, [])

  // Header 搜索框触发（HTTP 接口，结果也会通过 WS 广播回来）
  const handleSearchResults = (results) => {
    console.log('📊 Header 搜索结果 (HTTP):', results)
    // WS 广播会自动更新，这里可以不重复处理
    // 保留作为降级兼容
    setSearchResults(results)
  }

  return (
    <div className="app">
      <Header onSearchResults={handleSearchResults} />

      {/* 连接状态指示器 */}
      <div className="connection-status" style={{
        position: 'fixed',
        top: 10,
        right: 10,
        padding: '8px 12px',
        borderRadius: '4px',
        backgroundColor: (isConnected || apiAvailable) ? '#4CAF50' : '#f44336',
        color: 'white',
        fontSize: '12px',
        zIndex: 1000
      }}>
        {isConnected ? '✅ 实时连接' : apiAvailable ? '✅ HTTP 已连接' : '❌ 已断开'}
      </div>

      <main>
        <Routes>
          <Route
            path="/"
            element={
              <Home
                ws={ws}
                searchResults={searchResults}
                chatHistory={chatHistory}
                onSearchResultsClear={() => setSearchResults(null)}
              />
            }
          />
        </Routes>
      </main>
    </div>
  )
}

export default App
