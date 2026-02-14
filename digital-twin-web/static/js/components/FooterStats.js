/**
 * 底部统计栏组件
 */
app.component('footer-stats', {
  props: {
    systemStatus: Object,
    cameras: Array
  },
  
  computed: {
    onlineCameras() {
      return this.cameras.filter(c => c.status === 'online').length;
    },

    onlineRate() {
      const total = this.cameras.length || 0;
      if (total === 0) return 0;
      return Math.round((this.onlineCameras / total) * 100);
    }
  },
  
  template: `
    <div class="footer-stats">
      <div class="stat-item">
        <span class="stat-label">总摄像头:</span>
        <span class="stat-value">{{ cameras.length }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">在线:</span>
        <span class="stat-value">{{ onlineCameras }}</span>
      </div>
      <div class="stat-item stat-item--wide">
        <span class="stat-label">在线率:</span>
        <div class="stat-bar" :title="onlineRate + '%'"><div class="stat-bar-fill" :style="{ width: onlineRate + '%' }"></div></div>
        <span class="stat-value">{{ onlineRate }}%</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">今日告警:</span>
        <span class="stat-value">{{ systemStatus.total_alerts_today || 0 }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">运行时间:</span>
        <span class="stat-value">{{ systemStatus.system_uptime || '0分钟' }}</span>
      </div>
    </div>
  `
});
