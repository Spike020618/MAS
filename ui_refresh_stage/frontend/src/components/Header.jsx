import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.css';

export default function Header({ onSearchResults }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const onVoiceQuery = (event) => setSearchQuery(event.detail || '')
    window.addEventListener('media-agent:voice-query', onVoiceQuery)
    return () => window.removeEventListener('media-agent:voice-query', onVoiceQuery)
  }, [])

  // 从 .env 获取 API URL（通过 import.meta.env）
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/agent';

  const handleSearch = async (e) => {
    e?.preventDefault();
    
    if (!searchQuery.trim()) {
      alert('请输入搜索关键词');
      return;
    }
    
    setIsLoading(true);

    try {
      console.log(`🔍 搜索: ${searchQuery}`);
      
      // 使用正确的 API 地址
      const searchUrl = `${API_BASE_URL}/media_search?message=${encodeURIComponent(searchQuery)}`;
      console.log(`📡 请求 URL: ${searchUrl}`);
      
      const response = await fetch(searchUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('✅ 搜索结果:', data);

      // 处理搜索结果
      if (data.status === 'error') {
        alert(`❌ ${data.chat}`);
        return;
      }

      if (data.status === 'success') {
        // 向父组件传递搜索结果
        if (onSearchResults) {
          onSearchResults(data);
        }
        
        // 清空搜索框
        setSearchQuery('');
        
        // 显示成功提示
        console.log(`✅ 找到 ${data.medias?.length || 0} 个推荐`);
      }
    } catch (error) {
      console.error('❌ 搜索失败:', error);
      alert(`搜索失败: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <header className="header">
      <nav>
        <Link to="/" className="logo" aria-label="Media Agent home">
          <span className="mark">MA</span>
          <span className='name'>Media Agent</span>
        </Link>
        <form className="search" onSubmit={handleSearch}>
          <input
            className="search-input"
            type="text"
            placeholder="搜索视频、导演、演员..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            disabled={isLoading}
            aria-label="搜索视频"
          />
          <button 
            className="search-btn" 
            type="submit"
            disabled={isLoading}
            aria-label="搜索"
          >
            {isLoading ? '搜索中...' : '搜索'}
          </button>
        </form>
      </nav>
    </header>
  );
}
