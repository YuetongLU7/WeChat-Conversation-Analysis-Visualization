<script setup lang="ts">
import { ref, computed } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import type { UploadProps, UploadRawFile } from 'element-plus'
import { UploadFilled, Loading, Download } from '@element-plus/icons-vue'

const apiClient = axios.create({ baseURL: 'http://localhost:8082' });

const loading = ref(false)
const analysisResults = ref<any>(null)
const sentimentResults = ref<any>(null)
const emojiCloudData = ref<any>(null)
const reportUrl = ref<string | null>(null)

const uploadActionUrl = computed(() => `${apiClient.defaults.baseURL}/api/upload`)

const testBackend = async () => {
  try {
    const response = await apiClient.get('/api/test');
    ElMessage.success('Successfully connected to backend: ' + response.data.message);
  } catch (error: any) {
    const errorMsg = error.response?.data?.error || error.message;
    ElMessage.error('Connection failed: ' + errorMsg);
  }
}

const beforeUpload: UploadProps['beforeUpload'] = (rawFile: UploadRawFile) => {
  const isCSV = rawFile.type === 'text/csv' || rawFile.name.endsWith('.csv');
  if (!isCSV) {
    ElMessage.error('Only CSV files are allowed!');
    return false;
  }
  loading.value = true;
  analysisResults.value = null;
  sentimentResults.value = null;
  emojiCloudData.value = null;
  reportUrl.value = null;
  return true;
}

const handleUploadSuccess: UploadProps['onSuccess'] = (response: any) => {
  loading.value = false;
  if (response.success) {
    analysisResults.value = response.data;
    sentimentResults.value = response.sentiment;
    emojiCloudData.value = response.emoji_clouds || response.emoji_symbol_cloud;
    reportUrl.value = response.report_url;
    ElMessage.success('Analysis completed!');
  } else {
    ElMessage.error('Analysis failed: ' + (response.error || 'Unknown error'));
  }
}

const handleUploadError: UploadProps['onError'] = (err: any) => {
  loading.value = false;
  console.error('Upload error:', err);
  ElMessage.error('Upload failed, please try again!');
}

const getImageUrl = (path: string) => {
  if (!path) return '';
  return `${apiClient.defaults.baseURL}/${path}`;
}

const getWordCloudUrl = (path: string) => {
  if (!path) return '';
  return `${apiClient.defaults.baseURL}/${path}`;
}

const downloadReport = () => {
  if (reportUrl.value) {
    window.open(`${apiClient.defaults.baseURL}${reportUrl.value}`, '_blank');
  } else {
    ElMessage.error('Report not available');
  }
}
</script>

<template>
  <div class="home-container">
    <el-container>
      <el-header>
        <h1>WeChat Chat Sentiment Analysis</h1>
        <el-button @click="testBackend" type="primary" plain>Test Backend Connection</el-button>
      </el-header>
      
      <el-main>
        <el-row :gutter="20">
          <el-col :span="24">
            <el-card class="upload-card">
              <template #header><h2>Upload Chat Records</h2></template>
              <el-upload
                class="upload-demo"
                drag
                :action="uploadActionUrl"
                :on-success="handleUploadSuccess"
                :on-error="handleUploadError"
                :before-upload="beforeUpload"
                :show-file-list="false"
              >
                <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                <div class="el-upload__text">Drag file here or <em>click to upload</em></div>
                <template #tip><div class="el-upload__tip">Please upload CSV chat record file</div></template>
              </el-upload>
            </el-card>
          </el-col>
        </el-row>
        
        <el-row v-if="loading">
          <el-col :span="24" class="loading-container">
            <el-card>
              <div class="loading-text">
                <el-icon class="is-loading" size="40"><Loading /></el-icon>
                <p>Analyzing, please wait...</p>
              </div>
            </el-card>
          </el-col>
        </el-row>
        
        <div v-if="analysisResults" class="analysis-results">
          <el-row :gutter="20">
            <el-col :span="24">
              <el-card class="result-card">
                <template #header><h2>Time Analysis</h2></template>
                <el-row :gutter="20">
                  <el-col :xs="24" :sm="12">
                    <div class="stat-box">
                      <h3>Chat Time Span</h3>
                      <p>{{ analysisResults.time_analysis.first_date }} to {{ analysisResults.time_analysis.last_date }}</p>
                      <p>Total {{ analysisResults.time_analysis.total_days }} days</p>
                    </div>
                  </el-col>
                  <el-col :xs="24" :sm="12">
                    <div class="stat-box">
                      <h3>Message Statistics</h3>
                      <p>Total messages: {{ analysisResults.time_analysis.total_messages }}</p>
                      <p>Daily average: {{ (analysisResults.time_analysis.total_messages / analysisResults.time_analysis.total_days).toFixed(2) }} messages/day</p>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20" class="chart-row">
                  <el-col :xs="24" :md="12" class="chart-container">
                    <h3>Monthly Messages</h3>
                    <img :src="getImageUrl(analysisResults.time_analysis.monthly_chart)" class="chart-image" />
                  </el-col>
                  <el-col :xs="24" :md="12" class="chart-container">
                    <h3>Hourly Messages</h3>
                    <img :src="getImageUrl(analysisResults.time_analysis.hourly_chart)" class="chart-image" />
                  </el-col>
                </el-row>
                <el-row class="chart-row">
                  <el-col :span="24" class="chart-container">
                    <h3>Chat Frequency Heatmap</h3>
                    <img :src="getImageUrl(analysisResults.calendar_heatmap)" class="chart-image" />
                  </el-col>
                </el-row>
              </el-card>
            </el-col>
          </el-row>
          <el-row :gutter="20">
            <el-col :span="24">
              <el-card class="result-card">
                <template #header><h2>Word Cloud Analysis</h2></template>
                <el-tabs type="border-card">
                  <el-tab-pane label="Both"><iframe :src="getWordCloudUrl(analysisResults.word_cloud.both)" class="word-cloud-frame"></iframe></el-tab-pane>
                  <el-tab-pane label="Mine"><iframe :src="getWordCloudUrl(analysisResults.word_cloud.sender)" class="word-cloud-frame"></iframe></el-tab-pane>
                  <el-tab-pane label="Other"><iframe :src="getWordCloudUrl(analysisResults.word_cloud.receiver)" class="word-cloud-frame"></iframe></el-tab-pane>
                  <el-tab-pane label="French" v-if="analysisResults.word_cloud.french"><iframe :src="getWordCloudUrl(analysisResults.word_cloud.french)" class="word-cloud-frame"></iframe></el-tab-pane>
                </el-tabs>
              </el-card>
            </el-col>
          </el-row>
          <el-row :gutter="20" v-if="emojiCloudData && (emojiCloudData.sender || emojiCloudData.receiver)">
            <el-col :span="24">
              <el-card class="result-card">
                <template #header><h2>Emoji Usage Analysis</h2></template>
                <el-row :gutter="20">
                  <el-col :xs="24" :md="12" v-if="emojiCloudData.sender">
                    <h3 class="emoji-title">Your Emoji Usage</h3>
                    <iframe :src="getWordCloudUrl(emojiCloudData.sender)" class="emoji-cloud-frame"></iframe>
                  </el-col>
                  <el-col :xs="24" :md="12" v-if="emojiCloudData.receiver">
                    <h3 class="emoji-title">Other Person's Emoji Usage</h3>
                    <iframe :src="getWordCloudUrl(emojiCloudData.receiver)" class="emoji-cloud-frame"></iframe>
                  </el-col>
                </el-row>
              </el-card>
            </el-col>
          </el-row>
          <el-row :gutter="20" v-if="sentimentResults">
            <el-col :span="24">
              <el-card class="result-card">
                <template #header><h2>Sentiment Analysis</h2></template>
                <el-row :gutter="20">
                  <el-col :xs="24" :md="12" class="chart-container">
                    <h3>Emotion Radar Chart</h3>
                    <img :src="getImageUrl(sentimentResults.radar_chart)" class="chart-image" />
                  </el-col>
                  <el-col :xs="24" :md="12">
                    <div class="sentiment-summary">
                      <h3>Sentiment Summary</h3>
                      <p>{{ sentimentResults.summary }}</p>
                    </div>
                  </el-col>
                </el-row>
              </el-card>
            </el-col>
          </el-row>
          <el-row>
            <el-col :span="24" class="download-container">
              <el-button type="primary" size="large" @click="downloadReport">
                <el-icon class="el-icon--left"><Download /></el-icon>
                Download Full Report
              </el-button>
            </el-col>
          </el-row>
        </div>
      </el-main>

      <el-footer>
        <p>© 2025 WeChat Sentiment Analysis</p>
      </el-footer>
    </el-container>
  </div>
</template>

<style scoped>
.home-container, .el-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  width: 100%;
}

.el-main {
  flex-grow: 1;
  background-color: #f8f9fa;
  padding: 20px;
  width: 100%;
}

.el-header {
  background-color: #409EFF;
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  width: 100%;
}

.el-header h1 {
  margin: 0;
  font-size: 28px;
}

.el-footer {
  background-color: #545c64;
  color: #eee;
  padding: 20px;
  text-align: center;
  width: 100%;
}

.upload-card, .result-card {
  margin: 0 auto 20px auto;
  width: 100%;
}

.loading-container {
  max-width: 1200px;
  margin: 40px auto;
}
.loading-text {
  text-align: center;
  padding: 40px;
  color: #409EFF;
}
.loading-text .el-icon {
  font-size: 40px;
  margin-bottom: 20px;
}

.stat-box {
  background-color: #f5f7fa;
  border-radius: 4px;
  padding: 15px;
  margin-bottom: 20px;
  min-height: 120px;
  text-align: left;
}
.stat-box h3 {
  margin-top: 0;
}

.chart-row {
  margin-top: 20px;
}
.chart-container {
  margin-bottom: 20px;
}
.chart-image {
  max-width: 100%;
  border: 1px solid #ebeef5;
  border-radius: 4px;
}
.word-cloud-frame, .emoji-cloud-frame {
  width: 100%;
  height: 450px;
  border: none;
}
.emoji-title {
  text-align: center;
  margin-bottom: 15px;
  color: #303133;
}
.sentiment-summary {
  background-color: #f5f7fa;
  border-radius: 4px;
  padding: 20px;
  height: 100%;
  min-height: 300px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: left;
}
.download-container {
  margin: 30px 0;
  text-align: center;
}
</style>
