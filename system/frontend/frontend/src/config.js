// 前端配置文件
const config = {
  // API 基础 URL
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000/agent',

  // MCP 服务 URL
  mcpUrl: import.meta.env.VITE_MCP_URL || 'http://localhost:9000',

  // 前端 URL
  frontendUrl: import.meta.env.VITE_FRONTEND_URL || 'http://localhost:5173'
};

export default config;