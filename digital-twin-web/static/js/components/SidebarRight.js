/**
 * 右侧栏组件 - 告警列表
 */
app.component('sidebar-right', {
  props: {
    alerts: Array
  },
  
  methods: {
    getIcon(name) {
      const icons = {
        fire: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 22c4 0 7-3 7-7 0-5-4-6-4-10 0-2-1-3-3-5 0 3-2 4-3 6-1 2 0 3-2 5-2 2-3 3-3 6 0 4 3 5 8 5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 22c2.6 0 4.5-1.7 4.5-4.3 0-2.2-1.6-3.1-2.3-4.4-.6-1.2-.4-2.2-.7-3.3-1 .8-1.4 1.7-1.6 2.7-.2 1.2-.8 1.7-1.4 2.4-1 1-1.5 1.7-1.5 2.8 0 2.5 1.9 4.1 3 4.1Z" fill="currentColor" opacity="0.22"/></svg>',
        smoke: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7 19h10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M8 16c0-2 2-2 2-4s-2-2-2-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M12 16c0-2 2-2 2-4s-2-2-2-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M16 16c0-2 2-2 2-4s-2-2-2-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
        camera: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 8.5a2.5 2.5 0 0 1 2.5-2.5h5L13 7.5h4.5A2.5 2.5 0 0 1 20 10v7.5A2.5 2.5 0 0 1 17.5 20h-11A2.5 2.5 0 0 1 4 17.5V8.5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 17a3.2 3.2 0 1 0 0-6.4A3.2 3.2 0 0 0 12 17Z" stroke="currentColor" stroke-width="1.8"/></svg>'
      };
      return icons[name] || '';
    },

    onAlertClick(cameraId) {
      this.$emit('alert-click', cameraId);
    },
    
    getAlertClass(alert, index) {
      const classes = ['alert-item'];
      if (alert.type === 'smoke') classes.push('smoke');
      if (index === 0) classes.push('new');
      return classes.join(' ');
    },
    
    getAlertTypeText(type) {
      return type === 'fire' ? '火灾' : '烟雾';
    },
    
    formatTime(timestamp) {
      return new Date(timestamp).toLocaleTimeString('zh-CN');
    },
    
    formatConfidence(confidence) {
      return (confidence * 100).toFixed(0);
    }
  },
  
  template: `
    <div class="sidebar-right">
      <div class="sidebar-section">
        <div class="sidebar-section-title">告警记录</div>
        <div class="alert-list">
          <div 
            v-if="alerts.length === 0" 
            class="empty-state"
          >
            暂无告警
          </div>
          
          <div
            v-for="(alert, index) in alerts"
            :key="alert.id || index"
            :class="getAlertClass(alert, index)"
            @click="onAlertClick(alert.camera_id)"
          >
            <div class="alert-header">
              <span
                class="ui-icon"
                v-html="getIcon(alert.type === 'smoke' ? 'smoke' : 'fire')"
              ></span>
              <span
                :class="[
                  'badge',
                  alert.type === 'smoke' ? 'badge--warning' : 'badge--danger'
                ]"
              >
                {{ getAlertTypeText(alert.type) }}
              </span>
              <span class="alert-time">{{ formatTime(alert.timestamp) }}</span>
            </div>
            <div class="alert-camera">
              <span class="ui-icon muted-text" v-html="getIcon('camera')"></span>
              {{ alert.camera_name }}
            </div>
            <div class="alert-confidence">
              置信度: {{ formatConfidence(alert.confidence) }}%
            </div>
          </div>
        </div>
      </div>
    </div>
  `
});
