import axios from 'axios';
import { ElMessage } from 'element-plus';

/**
 * Create an Axios instance with a pre-configured base URL.
 * This is the central point for all API communications.
 */
const apiClient = axios.create({
  baseURL: 'http://localhost:8082/api',
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * A utility function to test the connection to the backend.
 * This can be called from any component.
 */
export const testBackendConnection = async () => {
  try {
    const response = await apiClient.get('/test');
    console.log('Backend response:', response.data);
    ElMessage.success('Successfully connected to backend: ' + response.data.message);
  } catch (error: any) {
    console.error('Failed to connect to backend:', error);
    const errorMsg = error.response?.data?.error || error.message;
    ElMessage.error('Connection failed: ' + errorMsg);
  }
};

export default apiClient; 