import config from '../config';

class ApiClient {
  constructor() {
    this.baseURL = config.apiUrl;
    this.mcpURL = config.mcpUrl;
  }

  // 通用请求方法
  async request(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : `${this.baseURL}${endpoint}`;
    
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      credentials: 'include', // 如果需要cookies
    };

    const response = await fetch(url, { ...defaultOptions, ...options });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({
        message: `HTTP error! status: ${response.status}`
      }));
      throw new Error(error.message || `Request failed with status ${response.status}`);
    }

    return response.json();
  }

  // GET请求
  get(endpoint, options = {}) {
    return this.request(endpoint, { method: 'GET', ...options });
  }

  // POST请求
  post(endpoint, data, options = {}) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
      ...options
    });
  }

  // PUT请求
  put(endpoint, data, options = {}) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
      ...options
    });
  }

  // DELETE请求
  delete(endpoint, options = {}) {
    return this.request(endpoint, { method: 'DELETE', ...options });
  }

  // MCP专用请求
  async mcpRequest(endpoint, data) {
    const url = `${this.mcpURL}${endpoint}`;
    return this.request(url, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }
}

// 创建单例实例
const apiClient = new ApiClient();
export default apiClient;