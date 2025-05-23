<template>
  <div class="home-view">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="upload-card">
          <div slot="header">
            <h2>Upload Chat Records</h2>
          </div>
          
          <el-upload
            class="upload-demo"
            drag
            :action="$http.defaults.baseURL + '/api/upload'"
            :headers="uploadHeaders"
            :on-success="handleUploadSuccess"
            :on-error="handleUploadError"
            :before-upload="beforeUpload"
            :show-file-list="false">
            <i class="el-icon-upload"></i>
            <div class="el-upload__text">
              Drag file here or <em>click to upload</em>
            </div>
            <div class="el-upload__tip" slot="tip">
              Please upload CSV chat record file exported from WeChat
            </div>
          </el-upload>
        </el-card>
      </el-col>
    </el-row>
    
    <!-- Loading indicator -->
    <el-row v-if="loading">
      <el-col :span="24" class="loading-container">
        <el-card>
          <div class="loading-text">
            <i class="el-icon-loading"></i>
            <p>Analyzing chat records, please wait...</p>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <!-- Analysis results -->
    <div v-if="analysisResults" class="analysis-results">
      <!-- Time Analysis Section -->
      <el-row :gutter="20">
        <el-col :span="24">
          <el-card class="result-card">
            <div slot="header">
              <h2>Time Analysis</h2>
            </div>
            
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
      
      <!-- Word Cloud Section -->
      <el-row :gutter="20">
        <el-col :span="24">
          <el-card class="result-card">
            <div slot="header">
              <h2>Word Cloud Analysis</h2>
            </div>
            
            <el-tabs type="border-card">
              <el-tab-pane label="Both">
                <iframe :src="getWordCloudUrl(analysisResults.word_cloud.both)" class="word-cloud-frame"></iframe>
              </el-tab-pane>
              
              <el-tab-pane label="Mine">
                <iframe :src="getWordCloudUrl(analysisResults.word_cloud.sender)" class="word-cloud-frame"></iframe>
              </el-tab-pane>
              
              <el-tab-pane label="Other">
                <iframe :src="getWordCloudUrl(analysisResults.word_cloud.receiver)" class="word-cloud-frame"></iframe>
              </el-tab-pane>
              
              <el-tab-pane label="French" v-if="analysisResults.word_cloud.french">
                <iframe :src="getWordCloudUrl(analysisResults.word_cloud.french)" class="word-cloud-frame"></iframe>
              </el-tab-pane>
            </el-tabs>
          </el-card>
        </el-col>
      </el-row>
      
      <!-- Emoji Cloud Section -->
      <el-row :gutter="20" v-if="emojiCloudData && (emojiCloudData.sender || emojiCloudData.receiver)">
        <el-col :span="24">
          <el-card class="result-card">
            <div slot="header">
              <h2>Emoji Usage Analysis</h2>
            </div>
            
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
      
      <!-- Sentiment Analysis Section -->
      <el-row :gutter="20">
        <el-col :span="24">
          <el-card class="result-card">
            <div slot="header">
              <h2>Sentiment Analysis</h2>
            </div>
            
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
      
      <!-- Download Report Button -->
      <el-row>
        <el-col :span="24" class="download-container">
          <el-button type="primary" size="large" @click="downloadReport" icon="el-icon-download">
            Download Full Report
          </el-button>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script>
export default {
  name: 'HomeView',
  data() {
    return {
      loading: false,
      analysisResults: null,
      sentimentResults: null,
      emojiCloudData: null,
      reportUrl: null,
      uploadHeaders: {}
    }
  },
  methods: {
    beforeUpload(file) {
      const isCSV = file.type === 'text/csv' || file.name.endsWith('.csv');
      if (!isCSV) {
        this.$message.error('Only CSV files are allowed!');
        return false;
      }
      
      this.loading = true;
      return true;
    },
    handleUploadSuccess(response) {
      this.loading = false;
      
      if (response.success) {
        this.analysisResults = response.data;
        this.sentimentResults = response.sentiment;
        this.emojiCloudData = response.emoji_clouds || response.emoji_symbol_cloud;
        this.reportUrl = response.report_url;
        
        this.$message.success('Analysis completed!');
      } else {
        this.$message.error('Analysis failed: ' + (response.error || 'Unknown error'));
      }
    },
    handleUploadError(err) {
      this.loading = false;
      console.error('Upload error:', err);
      this.$message.error('Upload failed, please try again!');
    },
    getImageUrl(path) {
      if (!path) return '';
      return `${this.$http.defaults.baseURL}/${path}`;
    },
    getWordCloudUrl(path) {
      if (!path) return '';
      return `${this.$http.defaults.baseURL}/${path}`;
    },
    downloadReport() {
      if (this.reportUrl) {
        window.open(`${this.$http.defaults.baseURL}${this.reportUrl}`, '_blank');
      } else {
        this.$message.error('Report not available');
      }
    }
  }
}
</script>

<style scoped>
.home-view {
  max-width: 1200px;
  margin: 0 auto;
}

.upload-card {
  margin-bottom: 20px;
}

.result-card {
  margin-bottom: 20px;
}

.loading-container {
  margin: 40px 0;
}

.loading-text {
  text-align: center;
  padding: 40px;
}

.loading-text i {
  font-size: 40px;
  margin-bottom: 20px;
  color: #409EFF;
}

.stat-box {
  background-color: #f5f7fa;
  border-radius: 4px;
  padding: 15px;
  margin-bottom: 20px;
  min-height: 120px;
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

.word-cloud-frame {
  width: 100%;
  height: 500px;
  border: none;
}

.emoji-cloud-frame {
  width: 100%;
  height: 400px;
  border: none;
  background-color: #f9f9f9;
  border-radius: 4px;
  overflow: hidden;
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
}

.download-container {
  margin: 30px 0;
  text-align: center;
}
</style> 