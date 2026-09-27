import { useState } from 'react';
import '../styles/MediaCard.css';

export default function MediaCard({ media }) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const placeholder = 'https://tse1-mm.cn.bing.net/th/id/OIP-C.dC6CLNcwmIg1I2fEbinJyQHaE5';

  // 处理媒体对象（兼容多种格式）
  const mediaData = {
    name: media?.name || '未知电影',
    director: media?.director || '未知导演',
    actors: media?.actors || '',
    genres: media?.genres || '',
    year: media?.year || '',
    rating: media?.rating || 0,
    description: media?.description || '',
    views: media?.views || '-',  // 播放量
    duration: media?.duration || '-',  // 时长
    id: media?.id || Math.random(),
    img: media?.img || '',
    url: media?.url || ''
  };

  // URL转换函数
  const transformMediaUrl = (url) => {
    if (!url) return '';
    
    // 如果是相对路径（以 / 开头），根据环境添加完整域名
    if (url.startsWith('/')) {
      const isProduction = window.location.hostname === 'www.tvosagent.cn' || window.location.hostname === 'tvosagent.cn';
      const protocol = window.location.protocol;
      const host = window.location.host;
      return `${protocol}//${host}${url}`;
    }
    
    // 如果是localhost:8000，保持原样
    if (url.includes('localhost:8000') || url.includes('127.0.0.1:8000')) {
      return url;
    }
    
    // 如果是完整的HTTP URL，尝试转为HTTPS
    if (url.startsWith('http://')) {
      return url.replace('http://', 'https://');
    }
    
    return url;
  };

  const mediaUrl = transformMediaUrl(mediaData.url);
  const mediaImg = mediaData.img ? transformMediaUrl(mediaData.img) : placeholder;

  return (
    <>
      <div className="video-card">
        <div className="thumbnail-container">
          <img 
            src={mediaImg}
            onError={(e) => { e.currentTarget.src = placeholder; }}
            className="thumbnail"
            alt={mediaData.name}
          />
          {/* 时长标签 */}
          {mediaData.duration && mediaData.duration !== '-' && (
            <span className="duration">{mediaData.duration}</span>
          )}
          <button 
            className="play-button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setIsFullscreen(true);
            }}
          >
            ▶
          </button>
        </div>
        <div className="info">
          <h3>{mediaData.name}</h3>
          
          {/* 导演信息 */}
          {mediaData.director && mediaData.director !== '未知导演' && (
            <div className="director">
              导演: {mediaData.director}
            </div>
          )}
          
          {/* 类型信息 */}
          {mediaData.genres && (
            <div className="genres">
              {mediaData.genres}
            </div>
          )}
          
          {/* 评分和年份 */}
          <div className="meta-info">
            {mediaData.rating > 0 && (
              <span className="rating">⭐ {mediaData.rating}</span>
            )}
            {mediaData.year && (
              <span className="year">{mediaData.year}年</span>
            )}
          </div>
          
          {/* 播放量 - 最重要的显示 */}
          {mediaData.views && mediaData.views !== '-' && (
            <div className="views-info">
              👁️ {mediaData.views}次观看
            </div>
          )}
          
          {/* 简介 */}
          {mediaData.description && (
            <div className="description">
              {mediaData.description.length > 40 
                ? mediaData.description.substring(0, 40) + '...' 
                : mediaData.description}
            </div>
          )}
        </div>
      </div>
      
      {/* 全屏播放层 */}
      {isFullscreen && (
        <div className="fullscreen-overlay">
          <button 
            className="close-button"
            onClick={() => setIsFullscreen(false)}
          >
            ×
          </button>
          <video 
            src={mediaUrl} 
            controls
            autoPlay
            allowFullScreen={true}
            style={{ 
              border: 'none',
              width: '100%',
              height: '100%'
            }}
          />
        </div>
      )}
    </>
  );
}
