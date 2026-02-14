/**
 * 传感器详情模态框组件
 */
app.component('sensor-modal', {
  props: {
    sensor: Object,
    facility: Object
  },
  
  computed: {
    sensorTypeName() {
      const typeMap = {
        'temperature_sensor': '温度传感器',
        'humidity_sensor': '湿度传感器',
        'smoke_detector': '烟雾探测器'
      };
      return typeMap[this.sensor?.type] || this.sensor?.type;
    },
    
    sensorIcon() {
      const icons = {
        'temperature_sensor': '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M14 14.8V5.5a2 2 0 1 0-4 0v9.3a3.5 3.5 0 1 0 4 0Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 17.5a1.2 1.2 0 1 0 0-2.4 1.2 1.2 0 0 0 0 2.4Z" fill="currentColor" opacity="0.85"/></svg>',
        'humidity_sensor': '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 3s6 6.3 6 11a6 6 0 1 1-12 0c0-4.7 6-11 6-11Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M9.5 14.5c.6 1.4 2 2.5 3.8 2.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
        'smoke_detector': '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7 19h10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M8 16c0-2 2-2 2-4s-2-2-2-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M12 16c0-2 2-2 2-4s-2-2-2-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M16 16c0-2 2-2 2-4s-2-2-2-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>'
      };
      return icons[this.sensor?.type] || '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 6h12v12H6V6Z" stroke="currentColor" stroke-width="1.8"/><path d="M9 9h6M9 12h6M9 15h4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
    },
    
    formattedValue() {
      if (!this.sensor || this.sensor.current_value === undefined) return '--';
      return typeof this.sensor.current_value === 'number' 
        ? this.sensor.current_value.toFixed(1) 
        : this.sensor.current_value;
    },
    
    formattedTimestamp() {
      if (!this.sensor?.timestamp) return '--';
      try {
        const date = new Date(this.sensor.timestamp);
        return date.toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        });
      } catch (e) {
        return this.sensor.timestamp;
      }
    },
    
    statusText() {
      return this.sensor?.status === 'alert' ? '告警' : '正常';
    },
    
    statusClass() {
      return this.sensor?.status === 'alert' ? 'status-alert' : 'status-normal';
    }
  },
  
  template: `
    <div class="modal-overlay" @click="$emit('close')">
      <div class="sensor-modal" @click.stop>
        <div class="modal-header">
          <div class="modal-title">
            <span class="ui-icon ui-icon--lg" v-html="sensorIcon"></span>
            <span>传感器详情</span>
          </div>
          <button class="modal-close" @click="$emit('close')">✕</button>
        </div>
        
        <div class="modal-body">
          <div class="sensor-info-row">
            <span class="info-label">名称</span>
            <span class="info-value">{{ facility?.name || sensor?.id }}</span>
          </div>
          
          <div class="sensor-info-row">
            <span class="info-label">类型</span>
            <span class="info-value">{{ sensorTypeName }}</span>
          </div>
          
          <div class="sensor-info-row">
            <span class="info-label">当前值</span>
            <span class="info-value value-highlight">
              {{ formattedValue }}{{ facility?.unit || '' }}
            </span>
          </div>
          
          <div class="sensor-info-row" v-if="sensor?.threshold">
            <span class="info-label">告警阈值</span>
            <span class="info-value">{{ sensor.threshold }}{{ facility?.unit || '' }}</span>
          </div>
          
          <div class="sensor-info-row">
            <span class="info-label">状态</span>
            <span :class="['info-value', 'status-badge', statusClass]">
              {{ statusText }}
            </span>
          </div>
          
          <div class="sensor-info-row" v-if="sensor?.timestamp">
            <span class="info-label">更新时间</span>
            <span class="info-value timestamp">{{ formattedTimestamp }}</span>
          </div>
        </div>
        
        <div class="modal-footer">
          <button class="btn-primary" @click="$emit('close')">确定</button>
        </div>
      </div>
    </div>
  `
});
