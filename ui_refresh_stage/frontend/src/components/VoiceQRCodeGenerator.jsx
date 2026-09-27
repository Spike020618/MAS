import React, { useRef, useState, useEffect } from 'react'
import QRCode from 'react-qr-code'
import '../styles/QRCodeGenerator.css'
import erIcon from '../assets/er.svg'

const VoiceQRCodeGenerator = () => {
  const qrCodeRef = useRef(null)
  const [currentUrl, setCurrentUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showQR, setShowQR] = useState(false)

  // 从 .env 获取 API URL
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/agent'

  const fetchServerIp = async () => {
    setIsLoading(true)
    try {
      // 调用后端获取 IP
      const response = await fetch(`${API_BASE_URL}/server_ip`)
      const data = await response.json()
      
      // 构建完整 URL（指向语音页面）
      const ip = data.ip || 'localhost'
      const voicePageUrl = `${API_BASE_URL}/voice_page`
      
      setCurrentUrl(voicePageUrl)
      console.log(`📱 语音页面 URL: ${voicePageUrl}`)
    } catch (error) {
      console.error('❌ 获取服务器信息失败:', error)
      // 降级方案
      setCurrentUrl(`${API_BASE_URL}/voice_page`)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchServerIp()
  }, [])

  return (
    <>
      {/* 固定在右上角的小开关 */}
      <button
        className={`qr-toggle-button ${showQR ? 'active' : ''}`}
        onClick={() => setShowQR(!showQR)}
        aria-expanded={showQR}
        title="扫描二维码进行语音搜索"
      >
        {showQR ? (
          <span>✕ 关闭</span>
        ) : (
          <span className="qr-btn-content">
            <img src={erIcon} alt="二维码" className="qr-btn-icon" />
            <span className="qr-btn-text">🎤 语音搜索</span>
          </span>
        )}
      </button>

      {/* 二维码模态框 */}
      {showQR && (
        <div className="qr-modal-overlay" onClick={() => setShowQR(false)}>
          <div className="qr-modal-content" onClick={(e) => e.stopPropagation()}>
            <button
              className="close-modal-button"
              onClick={() => setShowQR(false)}
              title="关闭"
            >
              ×
            </button>

            {isLoading ? (
              <div style={{ padding: '20px', textAlign: 'center' }}>
                ⏳ 加载中...
              </div>
            ) : (
              <>
                <h3 style={{ marginTop: '0', textAlign: 'center' }}>
                  🎤 扫描进行语音搜索
                </h3>
                
                {currentUrl && (
                  <div className="qrcode__image" ref={qrCodeRef}>
                    <QRCode
                      value={currentUrl}
                      size={280}
                      bgColor="#ffffff"
                      fgColor="#000000"
                      level="H"
                      includeMargin={true}
                    />
                  </div>
                )}
                
                <button 
                  onClick={fetchServerIp} 
                  className="refresh-button"
                  disabled={isLoading}
                >
                  🔄 刷新
                </button>
                
                <div className="current-url">
                  📍 URL: {currentUrl.replace('http://', '').replace('https://', '')}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  )
}

export default VoiceQRCodeGenerator
