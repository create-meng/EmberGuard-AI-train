/**
 * 左侧栏组件 - 摄像头列表和传感器面板
 */
app.component('sidebar-left', {
  props: {
    cameras: Array,
    sensors: Array
  },
  
  methods: {
    getIcon(name) {
      const icons = {
        camera: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 8.5a2.5 2.5 0 0 1 2.5-2.5h5L13 7.5h4.5A2.5 2.5 0 0 1 20 10v7.5A2.5 2.5 0 0 1 17.5 20h-11A2.5 2.5 0 0 1 4 17.5V8.5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 17a3.2 3.2 0 1 0 0-6.4A3.2 3.2 0 0 0 12 17Z" stroke="currentColor" stroke-width="1.8"/></svg>',
        temp: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M14 14.8V5.5a2 2 0 1 0-4 0v9.3a3.5 3.5 0 1 0 4 0Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 17.5a1.2 1.2 0 1 0 0-2.4 1.2 1.2 0 0 0 0 2.4Z" fill="currentColor" opacity="0.85"/></svg>',
        humidity: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 3s6 6.3 6 11a6 6 0 1 1-12 0c0-4.7 6-11 6-11Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M9.5 14.5c.6 1.4 2 2.5 3.8 2.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
        smoke: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7 19h10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M8 16c0-2 2-2 2-4s-2-2-2-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M12 16c0-2 2-2 2-4s-2-2-2-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M16 16c0-2 2-2 2-4s-2-2-2-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
        sensor: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 6h12v12H6V6Z" stroke="currentColor" stroke-width="1.8"/><path d="M9 9h6M9 12h6M9 15h4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>'
      };
      return icons[name] || '';
    },

    onCameraClick(cameraId, status) {
      if (status === 'offline') {
        alert('摄像头离线，请检查设备连接或启用演示模式');
        return;
      }
      this.$emit('camera-click', cameraId);
    },
    
    getCameraStatusClass(camera) {
      const classes = ['camera-item'];
      if (camera.status === 'offline') classes.push('offline');
      if (camera.alert_status === 'fire') classes.push('alert-fire');
      if (camera.alert_status === 'smoke') classes.push('alert-smoke');
      return classes.join(' ');
    },

    getCameraStatusText(camera) {
      if (camera.status === 'online') return '在线';
      if (camera.status === 'starting') return '启动中';
      return '离线';
    },

    getCameraDotClass(camera) {
      if (camera.status === 'online') return 'online';
      if (camera.status === 'starting') return 'starting';
      return 'offline';
    },

    getCameraBadgeClass(camera) {
      if (camera.status === 'online') return 'badge--success';
      if (camera.status === 'starting') return 'badge--warning';
      return 'badge--neutral';
    },
    
    getSensorIcon(type) {
      const icons = {
        'temperature_sensor': this.getIcon('temp'),
        'humidity_sensor': this.getIcon('humidity'),
        'smoke_detector': this.getIcon('smoke')
      };
      return icons[type] || this.getIcon('sensor');
    },
    
    getSensorValue(sensor) {
      if (!sensor.online || sensor.current_value === undefined || sensor.current_value === null) {
        return '--';
      }
      const value = typeof sensor.current_value === 'number' ? sensor.current_value.toFixed(1) : sensor.current_value;
      const unit = sensor.unit || '';
      return `${value}${unit}`;
    },
    
    getSensorClass(sensor) {
      const classes = ['sensor-item'];
      if (sensor.status === 'alert') classes.push('alert');
      return classes.join(' ');
    }
  },
  
  template: `
    <div class="sidebar-left">
      <!-- 摄像头列表 -->
      <div class="sidebar-section">
        <div class="sidebar-section-title">摄像头监控</div>
        <div class="camera-list grid-layout">
          <div 
            v-if="cameras.length === 0" 
            class="empty-state empty-state--grid"
          >
            暂无摄像头
          </div>
          
          <div
            v-for="(camera, index) in cameras"
            :key="camera.id"
            :class="getCameraStatusClass(camera)"
            @click="onCameraClick(camera.id, camera.status)"
          >
            <div class="camera-header">
              <span class="ui-icon" v-html="getIcon('camera')"></span>
              <span class="camera-name">{{ index + 1 }}</span>
            </div>
            <div class="camera-status">
              <span :class="['status-dot', getCameraDotClass(camera)]"></span>
              <span
                :class="['badge', getCameraBadgeClass(camera)]"
              >
                {{ getCameraStatusText(camera) }}
              </span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 传感器面板 -->
      <div class="sidebar-section">
        <div class="sidebar-section-title">传感器监控</div>
        <div class="sensor-list grid-layout">
          <div 
            v-if="sensors.length === 0" 
            class="empty-state empty-state--grid"
          >
            暂无传感器
          </div>
          
          <div
            v-for="(sensor, index) in sensors"
            :key="sensor.id"
            :class="getSensorClass(sensor)"
          >
            <div class="sensor-header">
              <span class="ui-icon" v-html="getSensorIcon(sensor.type)"></span>
              <span class="sensor-name">{{ index + 1 }}</span>
            </div>
            <div 
              :class="['sensor-value', sensor.status === 'alert' ? 'alert' : '']"
            >
              {{ getSensorValue(sensor) }}
            </div>
          </div>
        </div>
      </div>
    </div>
  `
});
