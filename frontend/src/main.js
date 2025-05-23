import Vue from 'vue'
import App from './App.vue'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import axios from 'axios'
import config from './config'

// Configure axios
axios.defaults.baseURL = config.API_BASE_URL
Vue.prototype.$http = axios

// Use Element UI
Vue.use(ElementUI)

Vue.config.productionTip = false

new Vue({
  render: h => h(App),
}).$mount('#app') 