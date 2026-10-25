import { useEffect, useState, useRef } from 'react'
import MediaCard from '../components/MediaCard'
import WebInputTools from '../components/WebInputTools'

// 来源标签映射
const SOURCE_LABEL = {
  media_search: '文字搜索',
  voice_page_search: '语音页面',
  chat: '对话',
}

const defaultMedias = [
  {
    id: '115713', name: 'Ex Machina', director: '',
    actors: '', genres: '剧情 · 科幻 · 惊悚',
    year: 2015, rating: 4.0, description: '人工智能、意识与人机关系主题的思想性科幻片',
    file_name: 'test.mp4', views: '1.2万', duration: '-',
    img: 'https://picsum.photos/640/360?random=21',
    url: '/agent/media/1'
  },
  {
    id: '106920', name: 'Her', director: '',
    actors: '', genres: '剧情 · 爱情 · 科幻',
    year: 2013, rating: 3.9, description: '从亲密关系与技术依赖切入的柔和科幻作品',
    file_name: 'test.mp4', views: '1.0万', duration: '-',
    img: 'https://picsum.photos/640/360?random=22',
    url: '/agent/media/1'
  },
  {
    id: '79132', name: 'Inception', director: '',
    actors: '', genres: '动作 · 剧情 · 科幻 · 悬疑',
    year: 2010, rating: 4.2, description: '以梦境结构、记忆与现实边界为核心的科幻悬疑片',
    file_name: 'test.mp4', views: '3.8万', duration: '-',
    img: 'https://picsum.photos/640/360?random=23',
    url: '/agent/media/1'
  },
  {
    id: '60069', name: 'WALL·E', director: '',
    actors: '', genres: '动画 · 科幻 · 爱情',
    year: 2008, rating: 4.0, description: '以机器人、环境与孤独为主题的动画科幻电影',
    file_name: 'test.mp4', views: '2.7万', duration: '-',
    img: 'https://picsum.photos/640/360?random=24',
    url: '/agent/media/1'
  }
]

export default function Home({ ws, searchResults, chatHistory = [], onSearchResultsClear }) {
  const [medias, setMedias] = useState(defaultMedias)
  const [displayTitle, setDisplayTitle] = useState('推荐视频')
  const [isSearchResult, setIsSearchResult] = useState(false)
  const [showTrace, setShowTrace] = useState(false)
  const [isChatOpen, setIsChatOpen] = useState(true)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [autoSpeak, setAutoSpeak] = useState(true)
  const [isMaleVoice, setIsMaleVoice] = useState(false)
  const [speed, setSpeed] = useState(1.0)
  
  const chatEndRef = useRef(null)
  const lastSpokenMsgIdRef = useRef(null)
  const speechSynthRef = useRef(null)
  const femaleVoiceRef = useRef(null)
  const maleVoiceRef = useRef(null)
  const voicesLoadedRef = useRef(false)

  // 初始化语音合成
  useEffect(() => {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      speechSynthRef.current = window.speechSynthesis
      
      const loadVoices = () => {
        const voices = speechSynthRef.current.getVoices()
        const chineseVoices = voices.filter(voice => 
          voice.lang.includes('zh-CN') || voice.lang.includes('zh_CN')
        )
        
        if (chineseVoices.length > 0) {
          femaleVoiceRef.current = chineseVoices.find(voice => 
            voice.name.includes('Tingting') || 
            voice.name.includes('Yaoyao') || 
            voice.name.includes('Meijia') ||
            voice.name.toLowerCase().includes('female')
          ) || chineseVoices[0]
          
          maleVoiceRef.current = chineseVoices.find(voice => 
            voice.name.includes('Kangkang') || 
            voice.name.includes('Yunyang') ||
            voice.name.toLowerCase().includes('male')
          ) || (chineseVoices[1] || chineseVoices[0])
          
          if (femaleVoiceRef.current === maleVoiceRef.current && chineseVoices.length > 1) {
            maleVoiceRef.current = chineseVoices[1]
          }
        } else if (voices.length > 0) {
          femaleVoiceRef.current = voices[0]
          maleVoiceRef.current = voices[0]
        }
        voicesLoadedRef.current = true
      }
      
      if (speechSynthRef.current.getVoices().length === 0) {
        speechSynthRef.current.onvoiceschanged = loadVoices
      } else {
        loadVoices()
      }
    }
    
    return () => {
      if (speechSynthRef.current) {
        speechSynthRef.current.cancel()
      }
    }
  }, [])

  // 自动播报新消息
  useEffect(() => {
    if (!autoSpeak) return
    if (!speechSynthRef.current) return
    if (!voicesLoadedRef.current) return
    if (chatHistory.length === 0) return

    const lastMsg = chatHistory[chatHistory.length - 1]
    
    if (lastMsg.role === 'assistant' && lastSpokenMsgIdRef.current !== lastMsg.id) {
      speakMessage(lastMsg.content)
      lastSpokenMsgIdRef.current = lastMsg.id
    }
  }, [chatHistory, autoSpeak])

  // 播报消息
  const speakMessage = (text) => {
    if (!speechSynthRef.current) return
    if (!voicesLoadedRef.current) return

    // 停止当前正在播放的语音
    speechSynthRef.current.cancel()

    const cleanText = text
      .replace(/[#*_`]/g, '')
      .replace(/\s+/g, ' ')
      .trim()

    if (!cleanText) return

    const utterance = new SpeechSynthesisUtterance(cleanText)
    utterance.lang = 'zh-CN'
    utterance.rate = speed
    utterance.pitch = isMaleVoice ? 0.8 : 1.2
    utterance.volume = 1
    
    if (isMaleVoice && maleVoiceRef.current) {
      utterance.voice = maleVoiceRef.current
    } else if (!isMaleVoice && femaleVoiceRef.current) {
      utterance.voice = femaleVoiceRef.current
    }

    utterance.onStart = () => setIsSpeaking(true)
    utterance.onEnd = () => setIsSpeaking(false)
    utterance.onError = () => setIsSpeaking(false)

    speechSynthRef.current.speak(utterance)
  }

  const stopSpeaking = () => {
    if (speechSynthRef.current) {
      speechSynthRef.current.cancel()
      setIsSpeaking(false)
    }
  }

  const speakLastMessage = () => {
    if (chatHistory.length === 0) return
    const lastMsg = chatHistory[chatHistory.length - 1]
    speakMessage(lastMsg.content)
  }

  // 清除聊天记录
  const clearChatHistory = () => {
    if (onSearchResultsClear) {
      onSearchResultsClear()
    }
    stopSpeaking()
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory])

  // 处理搜索结果
  useEffect(() => {
    if (!searchResults) return
    if (searchResults.type === 'chat') return

    if (searchResults.medias_info && searchResults.medias_info.length > 0) {
      setMedias(searchResults.medias_info)
      setDisplayTitle(`搜索结果 (${searchResults.medias_info.length}部影片)`)
      setIsSearchResult(true)
      setShowTrace(false)
    } else if (searchResults.medias && searchResults.medias.length > 0) {
      setMedias(searchResults.medias.map((name, idx) => ({
        id: idx,
        name,
        description: '推荐电影',
        views: '-',
        duration: '-',
        url: '/agent/media/1',
        file_name: 'test.mp4',
        img: `https://picsum.photos/640/360?random=${idx}`
      })))
      setDisplayTitle(`搜索结果 (${searchResults.medias.length}部影片)`)
      setIsSearchResult(true)
      setShowTrace(false)
    } else {
      setMedias([])
      setDisplayTitle('搜索结果 (0部影片)')
      setIsSearchResult(true)
    }

    setTimeout(() => {
      document.querySelector('.section-title')?.scrollIntoView({ behavior: 'smooth' })
    }, 300)
  }, [searchResults])

  const handleBackToRecommendations = () => {
    setMedias(defaultMedias)
    setDisplayTitle('推荐视频')
    setIsSearchResult(false)
    if (onSearchResultsClear) {
      onSearchResultsClear()
    }
  }

  const getSpeedLabel = () => {
    if (speed === 1.0) return '正常'
    if (speed === 1.5) return '较快'
    return '快速'
  }

  const toggleVoice = () => {
    setIsMaleVoice(!isMaleVoice)
    if (isSpeaking) {
      stopSpeaking()
    }
  }

  return (
    <div style={{ display: 'flex', gap: '0', minHeight: '100vh' }}>

      {/* 左侧主内容区 */}
      <div style={{ flex: 1, minWidth: 0 }} className="home-page">
        <WebInputTools />

        <div className="results-section">
          <h2 className="section-title" style={{ color: '#31404e' }}>
            {displayTitle}
          </h2>

          {isSearchResult && (
            <button
              className="back-button"
              onClick={handleBackToRecommendations}
              style={{
                marginBottom: '20px', padding: '8px 16px',
                backgroundColor: '#f44336', color: 'white',
                border: 'none', borderRadius: '4px',
                cursor: 'pointer', fontSize: '14px'
              }}
            >
              ← 返回推荐视频
            </button>
          )}

          {isSearchResult && searchResults?.trace && (
            <button onClick={() => setShowTrace(true)} style={{ marginBottom: '18px', padding: '9px 15px', border: '1px solid #b9cde8', borderRadius: '18px', background: '#f5f9ff', color: '#24558a', cursor: 'pointer', fontSize: '13px' }}>
              🔎 查看检索过程与共识轨迹
            </button>
          )}

          <div className="video-grid">
            {medias && medias.length > 0 ? (
              medias.map((media, i) => <MediaCard key={media.id ?? i} media={media} />)
            ) : (
              <div style={{ gridColumn: '1 / -1', padding: '40px 20px', textAlign: 'center', color: '#999' }}>
                {isSearchResult ? '未找到相关结果' : '加载推荐中...'}
              </div>
            )}
          </div>
        </div>
      </div>

      {showTrace && searchResults?.trace && (
        <div onClick={() => setShowTrace(false)} style={{ position: 'fixed', inset: 0, zIndex: 2000, background: 'rgba(15, 27, 48, .45)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
          <div onClick={(event) => event.stopPropagation()} style={{ width: 'min(680px, 100%)', maxHeight: '80vh', overflow: 'auto', background: '#fff', borderRadius: '14px', boxShadow: '0 18px 60px rgba(0,0,0,.25)', padding: '22px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <strong style={{ fontSize: '18px', color: '#24558a' }}>🔎 检索过程与共识轨迹</strong>
              <button onClick={() => setShowTrace(false)} aria-label="关闭检索轨迹" style={{ border: 0, background: '#eef3fa', borderRadius: '50%', width: '30px', height: '30px', cursor: 'pointer', fontSize: '18px' }}>×</button>
            </div>
            <div style={{ fontSize: '12px', color: '#718096', marginBottom: '12px' }}>机制：{searchResults.trace.mechanism}　拓扑：{searchResults.trace.topology || 'decentralized-logical'}</div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {(searchResults.trace.events || []).map((event, index) => <span key={`${event.event}-${index}`} style={{ padding: '6px 10px', borderRadius: '14px', background: '#f5f9ff', border: '1px solid #c8d9ef', color: '#345', fontSize: '12px' }}>{index + 1}. {({ planning: '规划', mechanism: '博弈契约', retrieval: '候选召回', deliberation: 'Agent审议', consensus: '共识排序' }[event.stage] || event.stage)}</span>)}
            </div>
            <div style={{ marginTop: '14px', paddingTop: '12px', borderTop: '1px solid #e7edf5', fontSize: '13px', color: '#5b6b83' }}>
              {searchResults.trace.reports?.length || 0} 条 Agent 报告 · 行动：{[...new Set((searchResults.trace.reports || []).map(report => report.action))].join('、') || '无'} · 证据与信誉共同参与最终排序
            </div>
            <details open style={{ marginTop: '14px', borderTop: '1px solid #e7edf5', paddingTop: '10px' }}>
              <summary style={{ cursor: 'pointer', color: '#24558a', fontWeight: 600 }}>规划参数与事件明细</summary>
              <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: '240px', overflow: 'auto', background: '#f7f9fc', padding: '10px', borderRadius: '7px', fontSize: '11px', color: '#4a5568' }}>{JSON.stringify({ plan: searchResults.trace.plan, events: searchResults.trace.events }, null, 2)}</pre>
            </details>
            <details open style={{ marginTop: '10px', borderTop: '1px solid #e7edf5', paddingTop: '10px' }}>
              <summary style={{ cursor: 'pointer', color: '#24558a', fontWeight: 600 }}>Agent 报告（逐条）</summary>
              <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: '300px', overflow: 'auto', background: '#f7f9fc', padding: '10px', borderRadius: '7px', fontSize: '11px', color: '#4a5568' }}>{JSON.stringify(searchResults.trace.reports || [], null, 2)}</pre>
            </details>
            <details open style={{ marginTop: '10px', borderTop: '1px solid #e7edf5', paddingTop: '10px' }}>
              <summary style={{ cursor: 'pointer', color: '#24558a', fontWeight: 600 }}>共识结果、异议与信誉变化</summary>
              <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: '300px', overflow: 'auto', background: '#f7f9fc', padding: '10px', borderRadius: '7px', fontSize: '11px', color: '#4a5568' }}>{JSON.stringify({ decisions: searchResults.trace.decisions || [], reputations_before: searchResults.trace.reputations_before || {}, reputations_after: searchResults.trace.reputations_after || {} }, null, 2)}</pre>
            </details>
          </div>
        </div>
      )}

      {/* 右侧聊天记录面板 */}
      <div style={{
        width: isChatOpen ? '320px' : '44px',
        flexShrink: 0,
        transition: 'width 0.3s ease',
        borderLeft: '1px solid #e0e0e0',
        background: '#f9fafb',
        display: 'flex',
        flexDirection: 'column',
        position: 'sticky',
        top: 0,
        height: '100vh',
        overflow: 'hidden'
      }}>
        {/* 标题栏 + 语音控制 */}
        <div style={{
          padding: '12px',
          borderBottom: '1px solid #e0e0e0',
          background: '#fff',
          flexShrink: 0
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: isChatOpen ? '12px' : 0
          }}>
            {isChatOpen && (
              <span style={{ fontWeight: 600, fontSize: '14px', color: '#333' }}>
                💬 聊天记录
                {chatHistory.length > 0 && (
                  <span style={{
                    marginLeft: '6px', background: '#1976d2', color: '#fff',
                    borderRadius: '10px', padding: '1px 7px', fontSize: '12px'
                  }}>
                    {chatHistory.length}
                  </span>
                )}
              </span>
            )}
            <button
              onClick={() => setIsChatOpen(v => !v)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                fontSize: '18px', color: '#555',
                marginLeft: isChatOpen ? '0' : 'auto',
                marginRight: isChatOpen ? '0' : 'auto'
              }}
            >
              {isChatOpen ? '›' : '‹'}
            </button>
          </div>

          {isChatOpen && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              flexWrap: 'wrap',
              fontSize: '12px'
            }}>
              {/* 自动播报开关 */}
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                <span style={{ color: '#666' }}>🔊</span>
                <input
                  type="checkbox"
                  checked={autoSpeak}
                  onChange={(e) => setAutoSpeak(e.target.checked)}
                  style={{ cursor: 'pointer', margin: 0 }}
                />
                <span style={{ color: '#666' }}>自动</span>
              </label>

              {/* 男声/女声切换 - 单个按钮 */}
              <button
                onClick={toggleVoice}
                style={{
                  padding: '2px 10px',
                  background: isMaleVoice ? '#2196f3' : '#e91e63',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  color: '#fff',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
              >
                🎤 {isMaleVoice ? '男声' : '女声'}
              </button>

              {/* 语速切换 */}
              <button
                onClick={() => setSpeed(speed === 1.0 ? 1.5 : speed === 1.5 ? 2.0 : 1.0)}
                style={{
                  padding: '2px 10px',
                  background: '#ff9800',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  color: '#fff',
                  minWidth: '46px'
                }}
              >
                ⚡ {getSpeedLabel()}
              </button>

              {/* 播报和控制按钮组 */}
              <div style={{ display: 'flex', gap: '6px', marginLeft: 'auto' }}>
                <button
                  onClick={speakLastMessage}
                  disabled={chatHistory.length === 0}
                  style={{
                    padding: '2px 10px',
                    background: chatHistory.length === 0 ? '#ccc' : '#1976d2',
                    border: 'none',
                    borderRadius: '12px',
                    cursor: chatHistory.length === 0 ? 'not-allowed' : 'pointer',
                    fontSize: '11px',
                    color: '#fff'
                  }}
                >
                  🔊 播报最新
                </button>
                {isSpeaking && (
                  <button
                    onClick={stopSpeaking}
                    style={{
                      padding: '2px 10px',
                      background: '#f44336',
                      border: 'none',
                      borderRadius: '12px',
                      cursor: 'pointer',
                      fontSize: '11px',
                      color: '#fff'
                    }}
                  >
                    ⏹️ 停止
                  </button>
                )}
                <button
                  onClick={clearChatHistory}
                  disabled={chatHistory.length === 0}
                  style={{
                    padding: '2px 10px',
                    background: chatHistory.length === 0 ? '#ccc' : '#666',
                    border: 'none',
                    borderRadius: '12px',
                    cursor: chatHistory.length === 0 ? 'not-allowed' : 'pointer',
                    fontSize: '11px',
                    color: '#fff'
                  }}
                  title="清除聊天记录"
                >
                  🗑️ 清空
                </button>
              </div>
            </div>
          )}
        </div>

        {/* 消息列表 */}
        {isChatOpen && (
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '12px 10px',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px'
          }}>
            {chatHistory.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#aaa', fontSize: '13px', marginTop: '40px' }}>
                <div style={{ fontSize: '28px', marginBottom: '8px' }}>💬</div>
                暂无聊天记录
              </div>
            ) : (
              chatHistory.map(msg => (
                <div
                  key={msg.id}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start'
                  }}
                >
                  <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '3px' }}>
                    {msg.role === 'user' ? '🧑 ' : '🤖 '}
                    {SOURCE_LABEL[msg.source] || msg.source} · {msg.timestamp}
                    {autoSpeak && msg.role === 'assistant' && lastSpokenMsgIdRef.current === msg.id && isSpeaking && (
                      <span style={{ marginLeft: '6px', color: '#4caf50' }}>🔊</span>
                    )}
                  </div>
                  <div style={{
                    maxWidth: '85%',
                    padding: '8px 12px',
                    borderRadius: msg.role === 'user' ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
                    background: msg.role === 'user' ? '#1976d2' : '#fff',
                    color: msg.role === 'user' ? '#fff' : '#333',
                    fontSize: '13px',
                    lineHeight: '1.5',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
                    wordBreak: 'break-word'
                  }}>
                    {msg.content}
                    {msg.hasMedia && (
                      <div style={{
                        marginTop: '5px',
                        fontSize: '12px',
                        color: msg.role === 'user' ? 'rgba(255,255,255,0.8)' : '#1976d2'
                      }}>
                        🎬 已更新 {msg.mediaCount} 部影视
                      </div>
                    )}
                  </div>
                  {/* 每条消息的单独播报按钮 - 只对AI回复显示 */}
                  {msg.role === 'assistant' && (
                    <button
                      onClick={() => speakMessage(msg.content)}
                      style={{
                        background: 'none',
                        border: 'none',
                        fontSize: '10px',
                        color: '#999',
                        cursor: 'pointer',
                        marginTop: '4px',
                        padding: '2px 8px',
                        borderRadius: '12px',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '3px',
                        transition: 'all 0.2s'
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.background = '#f0f0f0'
                        e.target.style.color = '#1976d2'
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.background = 'none'
                        e.target.style.color = '#999'
                      }}
                      title="播报此消息"
                    >
                      🔊 单独播报
                    </button>
                  )}
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>
        )}
      </div>

    </div>
  )
}
