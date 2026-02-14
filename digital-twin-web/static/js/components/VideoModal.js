/**
 * 视频弹窗组件
 */
app.component('video-modal', {
  props: {
    camera: Object
  },
  
  data() {
    return {
      videoFrame: null,
      lstmResult: null,
      updateInterval: null
    };
  },
  
  mounted() {
    this.syncFromCamera();
  },
  
  beforeUnmount() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  },

  watch: {
    camera: {
      deep: true,
      handler() {
        this.syncFromCamera();
      }
    }
  },
  
  computed: {
    lstmStatusClass() {
      if (this.camera && this.camera.status === 'offline') return 'offline';
      if (!this.lstmResult) return 'normal';
      
      if (this.lstmResult.lstm_prediction === 2) return 'fire';
      if (this.lstmResult.lstm_prediction === 1) return 'smoke';
      return 'normal';
    },
    
    lstmStatusText() {
      if (this.camera && this.camera.status === 'offline') return '离线';
      if (!this.lstmResult) return '无数据';
      return this.lstmResult.lstm_class_name || '正常';
    },
    
    lstmConfidence() {
      if (this.camera && this.camera.status === 'offline') return 0;
      if (!this.lstmResult) return 0;
      return (this.lstmResult.lstm_confidence * 100).toFixed(0);
    },
    
    probabilities() {
      if (!this.lstmResult || !this.lstmResult.lstm_probabilities) {
        return { normal: 0, smoke: 0, fire: 0 };
      }
      
      const probs = this.lstmResult.lstm_probabilities;
      return {
        normal: (probs['无火'] || 0) * 100,
        smoke: (probs['烟雾'] || 0) * 100,
        fire: (probs['火焰'] || 0) * 100
      };
    }
  },
  
  methods: {
    getIcon(name) {
      const icons = {
        camera: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 8.5a2.5 2.5 0 0 1 2.5-2.5h5L13 7.5h4.5A2.5 2.5 0 0 1 20 10v7.5A2.5 2.5 0 0 1 17.5 20h-11A2.5 2.5 0 0 1 4 17.5V8.5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 17a3.2 3.2 0 1 0 0-6.4A3.2 3.2 0 0 0 12 17Z" stroke="currentColor" stroke-width="1.8"/></svg>'
      };
      return icons[name] || '';
    },

    syncFromCamera() {
      if (!this.camera) return;

      if (this.camera.status === 'offline') {
        this.videoFrame = null; // 离线时清空画面
        this.lstmResult = null;
        return;
      }

      // 只有当有新画面或新检测结果时才更新
      if (this.camera.thumbnail) {
        this.videoFrame = this.camera.thumbnail;
      }

      if (this.camera.last_detection) {
        this.lstmResult = this.camera.last_detection;
      }
    },
    
    onClose() {
      this.$emit('close');
    }
  },
  
  template: `
    <div class="video-modal" @click.self="onClose">
      <div class="video-modal-content">
        <div class="video-modal-header">
          <div class="video-modal-title">
            <span class="ui-icon" v-html="getIcon('camera')"></span>
            <span>{{ camera.name }} - 实时监控</span>
          </div>
          <button class="video-modal-close" @click="onClose">×</button>
        </div>
        
        <div class="video-modal-body">
          <div class="video-display">
            <div class="video-frame">
              <img 
                v-if="videoFrame" 
                :src="videoFrame" 
                alt="视频加载中..." 
                class="video-frame-img"
              >
              <div v-else class="skeleton skeleton--frame" aria-label="视频加载中"></div>
            </div>
          </div>
          
          <div class="lstm-panel">
            <div class="lstm-result">
              <div :class="['lstm-status', lstmStatusClass]">
                {{ lstmStatusText }}
              </div>
              <div class="lstm-confidence">
                置信度: {{ lstmConfidence }}%
              </div>
            </div>
            
            <div class="lstm-probabilities">
              <div class="probability-bar">
                <div class="probability-label">
                  <span>无火</span>
                  <span>{{ probabilities.normal.toFixed(0) }}%</span>
                </div>
                <div class="probability-fill">
                  <div 
                    class="probability-value" 
                    :style="{ width: probabilities.normal + '%' }"
                  ></div>
                </div>
              </div>
              
              <div class="probability-bar">
                <div class="probability-label">
                  <span>烟雾</span>
                  <span>{{ probabilities.smoke.toFixed(0) }}%</span>
                </div>
                <div class="probability-fill">
                  <div 
                    class="probability-value smoke" 
                    :style="{ width: probabilities.smoke + '%' }"
                  ></div>
                </div>
              </div>
              
              <div class="probability-bar">
                <div class="probability-label">
                  <span>火焰</span>
                  <span>{{ probabilities.fire.toFixed(0) }}%</span>
                </div>
                <div class="probability-fill">
                  <div 
                    class="probability-value fire" 
                    :style="{ width: probabilities.fire + '%' }"
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `
});
