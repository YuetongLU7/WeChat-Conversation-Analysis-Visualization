/**
 * 全局配置文件
 */

// API基础URL，部署时修改为实际后端地址
const API_BASE_URL = process.env.VUE_APP_API_URL || 'http://localhost:8082';

// 导出配置
export default {
  API_BASE_URL
}; 