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
      updateInterval: null,
      lastDrawSig: null,
      preferVideo: true
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
      if (this.preferVideo && this.camera.demo_video_url) {
        this.videoFrame = this.camera.demo_video_url;
      } else if (this.camera.stream_url) {
        this.videoFrame = this.camera.stream_url;
      } else if (this.camera.thumbnail) {
        this.videoFrame = this.camera.thumbnail;
      }

      if (this.camera.last_detection) {
        this.lstmResult = this.camera.last_detection;
      }

      // 检测结果变化时触发绘制（避免高频循环绘制导致卡顿）
      const det = this.camera.last_detection;
      const sig = det ? JSON.stringify({
        t: det.timestamp,
        p: det.lstm_prediction,
        c: det.lstm_confidence,
        y: det.yolo_detections
      }) : '';
      if (sig !== this.lastDrawSig) {
        this.lastDrawSig = sig;
        this.$nextTick(() => this.drawOverlay());
      }
    },

    onVideoError() {
      // 浏览器不支持视频格式（例如 .avi）时，自动回退到 MJPEG。
      if (this.preferVideo) {
        this.preferVideo = false;
        this.$nextTick(() => {
          if (this.camera?.stream_url) {
            this.videoFrame = this.camera.stream_url;
          }
          this.syncFromCamera();
          this.drawOverlay();
        });
      }
    },

    drawOverlay() {
      const canvas = this.$refs.overlay;
      const media = this.$refs.videoEl || this.$refs.videoImg;
      if (!canvas || !media) return;

      const rect = media.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));

      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.clearRect(0, 0, w, h);

      const det = this.camera?.last_detection;
      const boxes = det?.yolo_detections || [];
      if (!Array.isArray(boxes) || boxes.length === 0) return;

      // 后端推理输入固定 640x480（DetectionEngine frame_resized）
      const srcW = 640;
      const srcH = 480;
      const sx = w / srcW;
      const sy = h / srcH;

      ctx.lineWidth = 2;
      ctx.font = '12px sans-serif';
      ctx.textBaseline = 'top';

      for (const b of boxes) {
        const bb = b?.bbox;
        if (!bb || bb.length !== 4) continue;
        const [x1, y1, x2, y2] = bb;
        const x = x1 * sx;
        const y = y1 * sy;
        const bw = (x2 - x1) * sx;
        const bh = (y2 - y1) * sy;

        const cls = b?.class_name || '';
        const conf = typeof b?.confidence === 'number' ? b.confidence : null;
        const label = conf === null ? cls : `${cls} ${(conf * 100).toFixed(0)}%`;

        const color = cls === 'fire' ? '#ff3b30' : '#ffcc00';
        ctx.strokeStyle = color;
        ctx.fillStyle = color;

        ctx.strokeRect(x, y, bw, bh);

        if (label) {
          const pad = 3;
          const metrics = ctx.measureText(label);
          const th = 14;
          const tw = Math.ceil(metrics.width) + pad * 2;
          const tx = Math.max(0, Math.min(w - tw, x));
          const ty = Math.max(0, y - th);
          ctx.fillRect(tx, ty, tw, th);
          ctx.fillStyle = '#000';
          ctx.fillText(label, tx + pad, ty + 1);
        }
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
              <video
                v-if="videoFrame && preferVideo"
                :src="videoFrame"
                class="video-frame-img"
                ref="videoEl"
                autoplay
                muted
                loop
                playsinline
                @loadedmetadata="drawOverlay"
                @error="onVideoError"
              ></video>
              <img
                v-else-if="videoFrame"
                :src="videoFrame"
                alt="视频加载中..."
                class="video-frame-img"
                ref="videoImg"
              >
              <canvas
                v-if="videoFrame"
                ref="overlay"
                style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;"
              ></canvas>
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
