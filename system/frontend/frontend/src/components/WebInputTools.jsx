import { useRef, useState } from 'react'
import QRCodeGenerator from './QRCodeGenerator'
import VoiceQRCodeGenerator from './VoiceQRCodeGenerator'

// 浏览器端输入能力：不再要求用户扫码到手机端。
export default function WebInputTools() {
  const fileRef = useRef(null)
  const [open, setOpen] = useState(false)
  const [listening, setListening] = useState(false)
  const [preview, setPreview] = useState('')

  const startVoice = () => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!Recognition) {
      window.alert('当前浏览器不支持网页语音输入，请直接使用顶部搜索框。')
      return
    }
    const recognition = new Recognition()
    recognition.lang = 'zh-CN'
    recognition.interimResults = false
    recognition.onstart = () => setListening(true)
    recognition.onend = () => setListening(false)
    recognition.onerror = () => setListening(false)
    recognition.onresult = (event) => {
      const text = event.results?.[0]?.[0]?.transcript?.trim()
      if (text) window.dispatchEvent(new CustomEvent('media-agent:voice-query', { detail: text }))
    }
    recognition.start()
  }

  const chooseImage = (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    setPreview(URL.createObjectURL(file))
  }

  return <div className="input-tools" style={{ margin: '0 0 14px' }}>
    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
      <QRCodeGenerator />
      <VoiceQRCodeGenerator />
      <button onClick={() => setOpen(value => !value)} aria-expanded={open} style={{ border: '1px solid #d7e1ed', borderRadius: '18px', padding: '7px 13px', background: '#fff', color: '#52657d', cursor: 'pointer', fontSize: '12px' }}>
        {open ? '⌃ 收起本机输入' : '⌄ 本机语音 / 图片'}
      </button>
    </div>
    {open && <div style={{ marginTop: '9px', display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
      <button onClick={startVoice} disabled={listening} style={{ border: 0, borderRadius: '16px', padding: '7px 12px', background: listening ? '#f0a44b' : '#e8f1ff', color: '#24558a', cursor: 'pointer', fontSize: '12px' }}>{listening ? '🎙️ 正在听…' : '🎙️ 网页语音输入'}</button>
      <button onClick={() => fileRef.current?.click()} style={{ border: 0, borderRadius: '16px', padding: '7px 12px', background: '#eef7ed', color: '#397044', cursor: 'pointer', fontSize: '12px' }}>🖼️ 选择图片</button>
      <input ref={fileRef} type="file" accept="image/*" onChange={chooseImage} hidden />
      {preview && <img src={preview} alt="已选择的图片" style={{ width: '58px', height: '38px', objectFit: 'cover', borderRadius: '5px', border: '1px solid #c8d5e5' }} />}
      <span style={{ fontSize: '11px', color: '#8a98aa' }}>本机图片仅预览；手机扫码入口可提交图片和文字描述</span>
    </div>}
  </div>
}
