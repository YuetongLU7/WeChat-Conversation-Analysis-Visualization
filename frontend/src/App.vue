<template>
  <div id="app">
    <el-button @click="testBackend">Test Backend Connection</el-button>
    <el-container>
      <el-header>
        <h1>WeChat Chat Sentiment Analysis</h1>
        <h2>WeChat Chat Sentiment Analysis</h2>
      </el-header>
      
      <el-main>
        <home-view />
      </el-main>
      
      <el-footer>
        <p>© 2025 WeChat Sentiment Analysis</p>
      </el-footer>
    </el-container>
  </div>
</template>

<script>
import HomeView from './views/HomeView.vue'

export default {
  name: 'App',
  components: {
    HomeView
  },
  methods: {
    async testBackend() {
      try {
        const response = await this.$http.get('/api/test');
        console.log('Backend response:', response.data);
        this.$message.success('Successfully connected to backend: ' + response.data.message);
      } catch (error) {
        console.error('Failed to connect to backend:', error);
        this.$message.error('Connection failed: ' + (error.response && error.response.data && error.response.data.error || error.message));
      }
    }
  }
}
</script>

<style>
#app {
  font-family: 'SimSun', 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin: 0;
  padding: 0;
  min-height: 100vh;
}

body {
  margin: 0;
  padding: 0;
  background-color: #f8f9fa;
}

.el-header {
  background-color: #409EFF;
  color: white;
  padding: 20px;
  text-align: center;
}

.el-header h1 {
  margin: 0;
  font-size: 28px;
}

.el-header h2 {
  margin: 5px 0 0 0;
  font-size: 18px;
  font-weight: normal;
}

.el-main {
  padding: 20px;
  min-height: calc(100vh - 180px);
}

.el-footer {
  background-color: #545c64;
  color: #eee;
  padding: 20px;
  text-align: center;
}
</style> 