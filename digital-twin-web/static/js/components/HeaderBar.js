/**
 * 顶部状态栏组件
 */
app.component('header-bar', {
  props: {
    systemStatus: Object,
    buildings: Array,
    currentBuilding: Object,
    floors: Array,
    currentFloor: Object
  },
  
  data() {
    return {
      currentTime: new Date(),
      alertEnabled: true,
      videoRecordingEnabled: false
    };
  },
  
  async mounted() {
    // 每秒更新时间
    setInterval(() => {
      this.currentTime = new Date();
    }, 1000);
    
    // 加载系统设置
    await this.loadSettings();
  },
  
  computed: {
    formattedTime() {
      return this.currentTime.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    },

    cameraOnlineText() {
      const online = this.systemStatus?.online_cameras ?? 0;
      const total = this.systemStatus?.total_cameras ?? 0;
      return `${online}/${total}`;
    }
  },
  
  methods: {
    getIcon(name) {
      const icons = {
        logo: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 10.5L12 4l9 6.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M6.5 10.5V20h11V10.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M10 20v-6h4v6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        bellOn: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M15 17H6c1.2-1.2 2-2.7 2-4.5V10a4 4 0 1 1 8 0v2.5c0 1.8.8 3.3 2 4.5h-3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M10 17a2 2 0 0 0 4 0" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        bellOff: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9.2 5.1A4 4 0 0 1 16 10v2.5c0 1.8.8 3.3 2 4.5H6c1.2-1.2 2-2.7 2-4.5V10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M10 17a2 2 0 0 0 4 0" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M4 4l16 16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
        recordOn: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="6" fill="currentColor" opacity="0.85"/><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.6" opacity="0.35"/></svg>',
        recordOff: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" opacity="0.85"/><rect x="4" y="4" width="16" height="16" rx="4" stroke="currentColor" stroke-width="1.6" opacity="0.35"/></svg>',
        camera: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 8.5a2.5 2.5 0 0 1 2.5-2.5h5L13 7.5h4.5A2.5 2.5 0 0 1 20 10v7.5A2.5 2.5 0 0 1 17.5 20h-11A2.5 2.5 0 0 1 4 17.5V8.5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 17a3.2 3.2 0 1 0 0-6.4A3.2 3.2 0 0 0 12 17Z" stroke="currentColor" stroke-width="1.8"/></svg>',
        alert: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 9v4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M12 17h.01" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/><path d="M10.2 4.8 2.6 18A2 2 0 0 0 4.3 21h15.4a2 2 0 0 0 1.7-3L13.8 4.8a2 2 0 0 0-3.6 0Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>',
        uptime: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 7v5l3 2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M21 12a9 9 0 1 1-3-6.7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>'
      };
      return icons[name] || '';
    },

    async loadSettings() {
      try {
        const response = await fetch('/api/system/settings');
        const result = await response.json();
        if (result.success) {
          this.alertEnabled = result.data.alert_enabled;
          this.videoRecordingEnabled = result.data.video_recording_enabled;
        }
      } catch (error) {
        console.error('加载系统设置失败:', error);
      }
    },
    
    async toggleAlert() {
      this.alertEnabled = !this.alertEnabled;
      await this.updateSettings();
    },
    
    async toggleVideoRecording() {
      this.videoRecordingEnabled = !this.videoRecordingEnabled;
      await this.updateSettings();
    },
    
    async updateSettings() {
      try {
        const response = await fetch('/api/system/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            alert_enabled: this.alertEnabled,
            video_recording_enabled: this.videoRecordingEnabled
          })
        });
        const result = await response.json();
        if (result.success) {
          console.log('系统设置已更新:', result.data);
        }
      } catch (error) {
        console.error('更新系统设置失败:', error);
      }
    },
    
    onBuildingChange(event) {
      this.$emit('building-change', event.target.value);
    },
    
    onFloorChange(event) {
      this.$emit('floor-change', event.target.value);
    }
  },
  
  template: `
    <div class="header-bar">
      <div class="header-left">
        <div class="header-logo">
          <span class="ui-icon ui-icon--accent" v-html="getIcon('logo')"></span>
          <span>数字孪生</span>
        </div>
        <div class="header-title">古建筑火灾监控系统</div>
      </div>
      
      <div class="header-center">
        <select 
          class="building-selector" 
          :value="currentBuilding?.id || ''"
          @change="onBuildingChange"
        >
          <option value="" disabled>请选择建筑</option>
          <option 
            v-for="building in buildings" 
            :key="building.id" 
            :value="building.id"
          >
            {{ building.name }}
          </option>
        </select>
        
        <select 
          v-if="floors.length > 0"
          class="floor-selector" 
          :value="currentFloor?.id || ''"
          @change="onFloorChange"
        >
          <option 
            v-for="floor in floors" 
            :key="floor.id" 
            :value="floor.id"
          >
            {{ floor.name }}
          </option>
        </select>
        
        <div class="system-controls">
          <button 
            class="control-btn" 
            :class="{ active: alertEnabled }"
            @click="toggleAlert"
            :title="alertEnabled ? '告警已启用' : '告警已禁用'"
          >
            <span class="ui-icon" v-html="getIcon(alertEnabled ? 'bellOn' : 'bellOff')"></span>
            告警
          </button>
          
          <button 
            class="control-btn" 
            :class="{ active: videoRecordingEnabled }"
            @click="toggleVideoRecording"
            :title="videoRecordingEnabled ? '视频录制已启用' : '视频录制已禁用'"
          >
            <span class="ui-icon" v-html="getIcon(videoRecordingEnabled ? 'recordOn' : 'recordOff')"></span>
            录制
          </button>
        </div>
      </div>
      
      <div class="header-right">
        <div class="header-pills">
          <div class="pill" :title="'在线/总摄像头'">
            <span class="ui-icon" v-html="getIcon('camera')"></span>
            <span class="pill-label">在线</span>
            <span class="pill-value">{{ cameraOnlineText }}</span>
          </div>
          <div class="pill pill--warning" :title="'今日告警'">
            <span class="ui-icon" v-html="getIcon('alert')"></span>
            <span class="pill-label">告警</span>
            <span class="pill-value">{{ systemStatus?.total_alerts_today || 0 }}</span>
          </div>
          <div class="pill" :title="'系统运行时间'">
            <span class="ui-icon" v-html="getIcon('uptime')"></span>
            <span class="pill-label">运行</span>
            <span class="pill-value">{{ systemStatus?.system_uptime || '0分钟' }}</span>
          </div>
        </div>
        <div class="header-time">{{ formattedTime }}</div>
      </div>
    </div>
  `
});
