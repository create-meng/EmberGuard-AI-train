/**
 * 数字孪生古建筑火灾监控系统 - Vue 3 主应用
 */

const { createApp } = Vue;

// API请求封装
const API = {
  async get(url) {
    try {
      const response = await fetch(url);
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API GET错误:', url, error);
      return { success: false, error: error.message };
    }
  },
  
  async post(url, body) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API POST错误:', url, error);
      return { success: false, error: error.message };
    }
  }
};

// 创建Vue应用实例（但不立即挂载）
const app = createApp({
  data() {
    return {
      // 系统状态
      systemStatus: {
        total_cameras: 0,
        online_cameras: 0,
        total_alerts_today: 0,
        system_uptime: '0分钟'
      },
      
      // 建筑和配置
      buildings: [],
      currentBuilding: null,
      currentFloor: null,
      floors: [],
      floorPlan: null,
      facilities: [],
      
      // 设备数据
      cameras: [],
      sensors: [],
      alerts: [],
      
      // UI状态
      showVideoModal: false,
      selectedCameraId: null,
      showSensorModal: false,
      selectedSensor: null,
      selectedFacility: null,
      loading: false,
      loadingMessage: '',
      wsConnected: false,
      cameraDataReady: false,
      sensorDataReady: false,
      lastCameraUpdateTs: 0,
      lastSensorUpdateTs: 0,
      
      // WebSocket
      socket: null,

      // Demo: SSE 结果流（低频）
      demoEventSource: null
    };
  },
  
  computed: {
    onlineCameras() {
      return this.cameras.filter(c => c.status === 'online').length;
    },

    selectedCamera() {
      if (!this.selectedCameraId) return null;
      const sid = String(this.selectedCameraId);
      return this.cameras.find(c => String(c.id) === sid) || null;
    }
  },
  
  async mounted() {
    console.log('Vue应用已挂载');

    this.startLoading('系统初始化中...');
    
    // 页面加载时，先清空后端的设备状态
    await API.post('/api/system/stop-all');
    
    // 监听页面刷新/关闭事件
    window.addEventListener('beforeunload', () => {
      // 使用 sendBeacon 发送异步请求（不会被浏览器阻止）
      navigator.sendBeacon('/api/system/stop-all');
    });
    
    await this.initApp();
  },
  
  methods: {
    stopDemoEventSource() {
      if (this.demoEventSource) {
        try {
          this.demoEventSource.close();
        } catch (e) {
          // ignore
        }
        this.demoEventSource = null;
      }
    },

    startDemoEventSource(cameraId) {
      this.stopDemoEventSource();
      const cid = String(cameraId);
      const url = `/demo/events?camera_id=${encodeURIComponent(cid)}`;

      try {
        const es = new EventSource(url);
        this.demoEventSource = es;

        es.onmessage = (evt) => {
          if (!evt?.data) return;
          let payload;
          try {
            payload = JSON.parse(evt.data);
          } catch (e) {
            return;
          }

          // 摄像头检测结果（轻量）
          const camPayload = payload?.camera;
          if (camPayload?.camera_id) {
            const pcid = String(camPayload.camera_id);
            const camera = this.cameras.find(c => String(c.id) === pcid);
            if (camera) {
              this.lastCameraUpdateTs = Date.now();
              this.cameraDataReady = true;
              if (camPayload.status) camera.status = camPayload.status;
              if (camPayload.last_detection) {
                camera.last_detection = camPayload.last_detection;
                this.normalizeCameraAlertStatus(camera, camera.last_detection);
              }
            }
          }

          // 传感器快照（低频合并更新）
          const sensorsPayload = payload?.sensors;
          if (Array.isArray(sensorsPayload)) {
            this.lastSensorUpdateTs = Date.now();
            this.sensorDataReady = true;
            if (!Array.isArray(this.sensors) || this.sensors.length === 0) {
              this.sensors = sensorsPayload;
            } else {
              for (const s of sensorsPayload) {
                const existing = this.sensors.find(x => x.id === s.id);
                if (existing) Object.assign(existing, s);
              }
            }
          }
        };

        es.onerror = () => {
          // SSE 失败时不弹窗刷屏，保持静默
        };
      } catch (e) {
        // ignore
      }
    },
    startLoading(message) {
      this.loadingMessage = message || '加载中...';
      this.loading = true;
    },

    stopLoading() {
      this.loading = false;
      this.loadingMessage = '';
    },

    resetRealtimeReadiness() {
      this.wsConnected = !!(this.socket && this.socket.connected);
      this.cameraDataReady = false;
      this.sensorDataReady = false;
      this.lastCameraUpdateTs = 0;
      this.lastSensorUpdateTs = 0;
    },

    prepareRealtimeReadinessForCurrentData() {
      if (this.cameras.length === 0) this.cameraDataReady = true;
      if (this.sensors.length === 0) this.sensorDataReady = true;
    },

    shouldWaitForRealtimeReady() {
      if (!this.currentBuilding) return false;
      return true;
    },

    async waitForRealtimeReady({ timeoutMs = 8000 } = {}) {
      const startedAt = Date.now();
      return new Promise(resolve => {
        const timer = setInterval(() => {
          const elapsed = Date.now() - startedAt;
          const ok = this.wsConnected && this.cameraDataReady && this.sensorDataReady;

          if (ok) {
            clearInterval(timer);
            resolve({ ready: true, timeout: false });
            return;
          }

          if (elapsed >= timeoutMs) {
            clearInterval(timer);
            resolve({ ready: false, timeout: true });
          }
        }, 150);
      });
    },

    normalizeCameraAlertStatus(camera, lastDetection) {
      if (camera && camera.status === 'offline') {
        camera.alert_status = 'offline';
        return;
      }
      if (!lastDetection || typeof lastDetection !== 'object') return;
      const pred = lastDetection.lstm_prediction;
      if (pred === 2) camera.alert_status = 'fire';
      else if (pred === 1) camera.alert_status = 'smoke';
      else if (pred === 0) camera.alert_status = 'normal';
    },

    // 初始化应用
    async initApp() {
      console.log('初始化应用...');
      
      // 加载数据
      await this.loadSystemConfig();
      await this.loadBuildings();
      await this.loadCameras();
      await this.loadSensors();
      await this.loadAlerts();
      
      // 初始化WebSocket
      this.resetRealtimeReadiness();
      this.initWebSocket();

      this.prepareRealtimeReadinessForCurrentData();

      if (!this.shouldWaitForRealtimeReady()) {
        this.stopLoading();
        console.log('未选择建筑，跳过实时就绪等待');
        // 启动定时更新
        this.startPeriodicUpdates();
        console.log('应用初始化完成');
        return;
      }

      const readyResult = await this.waitForRealtimeReady({ timeoutMs: 8000 });
      if (readyResult.timeout) {
        const offlineCameras = this.cameras.filter(c => c.status === 'offline');
        if (this.cameras.length > 0 && offlineCameras.length > 0) {
          this.showNotification(`部分摄像头异常/离线（${offlineCameras.length}）`, 'info');
        } else {
          this.showNotification('实时数据初始化超时，已进入演示界面', 'info');
        }
      }
      this.stopLoading();
      
      // 启动定时更新
      this.startPeriodicUpdates();
      
      console.log('应用初始化完成');
    },
    
    // 加载系统配置
    async loadSystemConfig() {
      const result = await API.get('/api/config');
      if (result.success) {
        this.facilities = result.data.facilities || [];
        this.floorPlan = result.data.floor_plan;
        console.log('系统配置加载完成');
      }
    },
    
    // 加载建筑列表
    async loadBuildings() {
      const result = await API.get('/api/buildings');
      if (result.success) {
        this.buildings = result.data;
        // 不自动选择建筑，让用户手动选择
        this.currentBuilding = null;
        console.log('建筑列表加载完成:', this.buildings.length);
      }
    },
    
    // 加载摄像头列表
    async loadCameras() {
      // 如果没有选中建筑，不加载摄像头
      if (!this.currentBuilding) {
        this.cameras = [];
        return;
      }
      
      const result = await API.get('/api/cameras');
      if (result.success) {
        const normalized = (result.data || []).map(c => ({ ...c, id: String(c.id) }));
        // 如果有选中楼层，只显示当前楼层的摄像头
        if (this.currentFloor && this.facilities.length > 0) {
          const floorCameraIds = this.facilities
            .filter(f => f.type === 'camera')
            .map(f => String(f.id));
          this.cameras = normalized.filter(c => floorCameraIds.includes(String(c.id)));
        } else {
          this.cameras = normalized;
        }
        console.log('摄像头列表加载完成:', this.cameras.length);
      }
    },
    
    // 加载传感器列表
    async loadSensors() {
      // 如果没有选中建筑，不加载传感器
      if (!this.currentBuilding) {
        this.sensors = [];
        return;
      }
      
      const result = await API.get('/api/sensors');
      if (result.success) {
        // 如果有选中楼层，只显示当前楼层的传感器
        if (this.currentFloor && this.facilities.length > 0) {
          const floorSensorIds = this.facilities
            .filter(f => ['temperature_sensor', 'humidity_sensor', 'smoke_detector'].includes(f.type))
            .map(f => f.id);
          this.sensors = result.data.filter(s => floorSensorIds.includes(s.id));
        } else {
          this.sensors = result.data;
        }
        console.log('传感器列表加载完成:', this.sensors.length);
      }
    },
    
    // 加载告警列表
    async loadAlerts() {
      const result = await API.get('/api/alerts?limit=10');
      if (result.success) {
        this.alerts = result.data;
        console.log('告警列表加载完成:', this.alerts.length);
      }
    },
    
    // 初始化WebSocket
    initWebSocket() {
      console.log('初始化 WebSocket 连接...');
      
      // 检查 Socket.IO 是否加载
      if (typeof io === 'undefined') {
        console.error('✗ Socket.IO 客户端未加载！');
        return;
      }
      
      // 连接 Socket.IO 
      this.socket = io({
        transports: ['websocket', 'polling'],
        upgrade: true,
        rememberUpgrade: true
      });

      this.socket.on('connect', () => {
        console.log('✔ WebSocket 连接成功, ID:', this.socket.id);
        this.wsConnected = true;
      });

      this.socket.on('disconnect', () => {
        console.log('✗ WebSocket 连接断开');
        this.wsConnected = false;
      });
      
      this.socket.on('connect_error', (error) => {
        console.error('✗ WebSocket 连接错误:', error);
      });
      
      // 摄像头更新事件
      this.socket.on('camera_update', (data) => {
        this.handleCameraUpdate(data);
      });

      // 启动视频回执（返回 stream_url）
      this.socket.on('video_started', (data) => {
        this.handleVideoStarted(data);
      });

      // 视频帧事件
      this.socket.on('video_frame', (data) => {
        console.warn(`[SOCKET] 收到 video_frame | camera: ${data?.camera_id} | thumbnail: ${!!data?.thumbnail} | size: ${data?.thumbnail ? data.thumbnail.length : 0}`);
        this.handleVideoFrame(data);
      });
      
      // 新告警事件
      this.socket.on('new_alert', (alert) => {
        console.log('收到新告警:', alert);
        this.handleNewAlert(alert);
      });
      
      // 传感器更新事件
      this.socket.on('sensor_update', (data) => {
        this.handleSensorUpdate(data);
      });
      
      // 传感器告警事件
      this.socket.on('sensor_alert', (alert) => {
        this.handleSensorAlert(alert);
      });
      
      // 配置文件变化事件
      this.socket.on('config_changed', (data) => {
        console.log('✓ 收到配置变化通知:', data.file);
        this.handleConfigChanged(data);
      });
    },
    
    // 处理摄像头更新
    handleCameraUpdate(data) {
      const cid = String(data.camera_id);
      const camera = this.cameras.find(c => String(c.id) === cid);
      if (camera) {
        this.lastCameraUpdateTs = Date.now();
        this.cameraDataReady = true;
        
        // 状态真正变化时才更新
        if (data.status && data.status !== camera.status) {
           camera.status = data.status;
        }
        
        camera.fps = data.fps;

        // 如果摄像头离线，清空数据
        if (camera.status === 'offline') {
          camera.last_detection = null;
          camera.thumbnail = null;
        }

        // 如果当前正在观看此摄像头，且 camera_update 里没带 thumbnail，
        // 则不要覆盖 handleVideoFrame 正在更新的画面数据
        this.normalizeCameraAlertStatus(camera, camera.last_detection);
      }
    },

    // 处理订阅摄像头的视频帧
    handleVideoFrame(data) {
      const cid = String(data.camera_id);
      const camera = this.cameras.find(c => String(c.id) === cid);
      if (!camera) return;

      Object.assign(camera, {
        status: data.status,
        last_detection: data.last_detection,
        thumbnail: data.thumbnail
      });

      if (camera.status === 'offline') {
        camera.last_detection = null;
        camera.thumbnail = null;
      }

      this.normalizeCameraAlertStatus(camera, camera.last_detection);
    },

    // 处理 start_video 的回执：绑定 MJPEG 流地址
    handleVideoStarted(data) {
      if (!data || !data.success) return;
      const cid = String(data.camera_id);
      if (!this.showVideoModal) return;
      if (String(this.selectedCameraId) !== cid) return;
      const camera = this.cameras.find(c => String(c.id) === cid);
      if (!camera) return;

      // 后端返回的是相对路径 /stream/<camera_id>
      camera.stream_url = data.stream_url || `/stream/${cid}`;
    },
    
    // 处理新告警
    handleNewAlert(alert) {
      console.log('收到新告警:', alert);
      
      // 添加到告警列表顶部
      this.alerts.unshift(alert);
      if (this.alerts.length > 10) {
        this.alerts = this.alerts.slice(0, 10);
      }
      
      // 摄像头告警状态统一由 camera_update.last_detection 派生，避免多处覆盖
      
      // 播放告警音效
      this.playAlertSound();
    },
    
    // 处理传感器更新
    handleSensorUpdate(data) {
      const sensor = this.sensors.find(s => s.id === data.sensor_id);
      if (sensor) {
        this.lastSensorUpdateTs = Date.now();
        this.sensorDataReady = true;
        Object.assign(sensor, {
          current_value: data.value,
          status: data.status,
          timestamp: data.timestamp
        });
      }
    },
    
    // 处理传感器告警
    handleSensorAlert(alert) {
      // 静默处理
    },
    
    // 处理配置文件变化
    async handleConfigChanged(data) {
      console.log('处理配置变化, 当前建筑:', this.currentBuilding);
      
      // 如果当前有选中的建筑，自动重新加载
      if (this.currentBuilding) {
        console.log('开始重新加载配置...');
        
        // 延迟一下，确保文件写入完成
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // 重新切换建筑（会重新加载配置）
        await this.handleBuildingChange(this.currentBuilding.id);
        
        // 显示提示
        this.showNotification('配置已更新', 'success');
        
        console.log('配置重新加载完成');
      } else {
        console.log('没有选中建筑，跳过重新加载');
      }
    },
    
    // 显示通知
    showNotification(message, type = 'info') {
      // 简单的通知实现
      const notification = document.createElement('div');
      notification.className = `notification notification-${type}`;
      notification.textContent = message;
      notification.style.animation = 'slideInRight 0.3s ease-out';
      
      document.body.appendChild(notification);
      
      // 3秒后移除
      setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
      }, 3000);
    },
    
    // 播放告警音效
    playAlertSound() {
      try {
        const audio = new Audio('/static/sounds/alert.mp3');
        audio.volume = 0.5;
        audio.play().catch(err => {
          console.log('音效播放失败（可能被浏览器阻止）');
        });
      } catch (error) {
        console.log('音效文件不存在');
      }
    },
    
    // 处理建筑切换
    async handleBuildingChange(buildingId) {
      if (!buildingId) {
        this.currentBuilding = null;
        this.currentFloor = null;
        this.floors = [];
        this.floorPlan = null;
        this.facilities = [];
        this.cameras = [];
        this.sensors = [];
        return;
      }
      
      console.log('切换建筑:', buildingId);
      
      this.startLoading('正在切换建筑...');
      
      // 记录开始时间
      const startTime = Date.now();
      
      try {
        const result = await API.post('/api/buildings/switch', {
          building_id: buildingId
        });
        
        if (result.success) {
          // 切换成功，加载建筑详细配置
          const building = this.buildings.find(b => b.id === buildingId);
          this.currentBuilding = building;
          
          // 加载建筑详细配置（包含楼层）
          const buildingDetail = await API.get(`/api/buildings/${buildingId}`);
          if (buildingDetail.success) {
            const config = buildingDetail.data;
            
            console.log('建筑配置:', config);
            
            // 保存楼层列表
            this.floors = config.floors || [];
            
            console.log('楼层列表:', this.floors);
            
            // 默认选择第一个楼层（跳过加载动画，因为建筑切换已经有了）
            if (this.floors.length > 0) {
              await this.switchFloor(this.floors[0].id, true);
            } else {
              // 兼容旧格式（没有楼层）
              this.floorPlan = config.floor_plan;
              this.facilities = config.facilities || [];

              // 重新加载设施数据（旧格式才需要）
              await this.loadCameras();
              await this.loadSensors();
            }
          }

          // 只在有建筑时等待实时更新
          this.resetRealtimeReadiness();
          this.prepareRealtimeReadinessForCurrentData();
          if (this.shouldWaitForRealtimeReady()) {
            const readyResult = await this.waitForRealtimeReady({ timeoutMs: 6000 });
            if (readyResult.timeout) {
              const offlineCameras = this.cameras.filter(c => c.status === 'offline');
              if (this.cameras.length > 0 && offlineCameras.length > 0) {
                this.showNotification(`部分摄像头异常/离线（${offlineCameras.length}）`, 'info');
              }
            }
          }
          
          console.log('✓ 建筑切换成功');
        } else {
          alert('切换失败: ' + result.error);
        }
      } catch (error) {
        alert('切换失败: ' + error.message);
      } finally {
        // 确保至少显示1秒的加载动画
        const elapsed = Date.now() - startTime;
        const remainingTime = Math.max(0, 1000 - elapsed);
        
        await new Promise(resolve => setTimeout(resolve, remainingTime));
        
        this.stopLoading();
      }
    },
    
    // 切换楼层
    async switchFloor(floorId, skipLoading = false) {
      const floor = this.floors.find(f => f.id === floorId);
      if (!floor) {
        console.error('未找到楼层配置:', floorId);
        return;
      }
      
      console.log('切换楼层:', floor.name);
      
      // 如果不跳过加载动画
      if (!skipLoading) {
        this.startLoading(`正在切换到${floor.name}...`);
      }
      
      // 记录开始时间
      const startTime = Date.now();
      
      try {
        // 调用后端API切换楼层
        const result = await API.post(`/api/buildings/${this.currentBuilding.id}/floors/${floorId}/switch`);
        
        if (result.success) {
          // 更新前端状态
          this.currentFloor = floor;
          this.floorPlan = floor.floor_plan;
          
          // 彻底清空并重置设施列表，触发子组件重新渲染
          this.facilities = [];
          await this.$nextTick(); 
          this.facilities = floor.facilities || [];
          
          // 重新加载设备数据
          await this.loadCameras();
          await this.loadSensors();

          this.resetRealtimeReadiness();
          this.prepareRealtimeReadinessForCurrentData();
          if (this.shouldWaitForRealtimeReady()) {
            const readyResult = await this.waitForRealtimeReady({ timeoutMs: 6000 });
            if (readyResult.timeout) {
              const offlineCameras = this.cameras.filter(c => c.status === 'offline');
              if (this.cameras.length > 0 && offlineCameras.length > 0) {
                this.showNotification(`部分摄像头异常/离线（${offlineCameras.length}）`, 'info');
              }
            }
          }
          
          console.log('✓ 楼层切换成功:', result.message);
        } else {
          console.error('楼层切换接口返回失败:', result.error);
          this.showNotification('楼层切换失败: ' + (result.error || '未知错误'), 'error');
        }
      } catch (error) {
        console.error('楼层切换异常:', error);
        this.showNotification('楼层切换异常: ' + error.message, 'error');
      } finally {
        // 如果不跳过加载动画
        if (!skipLoading) {
          // 确保至少显示1秒的加载动画
          const elapsed = Date.now() - startTime;
          const remainingTime = Math.max(0, 1000 - elapsed);
          
          await new Promise(resolve => setTimeout(resolve, remainingTime));
          
          this.stopLoading();
        }
      }
    },
    
    // 处理设施点击
    handleFacilityClick(facility) {
      console.log('点击设施:', facility);
      
      if (facility.type === 'camera') {
        this.openVideoModal(facility.id);
      } else if (facility.type === 'temperature_sensor' || facility.type === 'humidity_sensor' || facility.type === 'smoke_detector') {
        this.showSensorDetail(facility.id);
      }
    },
    
    // 处理告警点击
    handleAlertClick(cameraId) {
      this.openVideoModal(cameraId);
    },
    
    // 打开视频弹窗
    openVideoModal(cameraId) {
      const cid = String(cameraId);
      const camera = this.cameras.find(c => String(c.id) === cid);
      if (!camera) return;
      
      if (camera.status === 'offline') {
        alert('摄像头离线，请检查设备连接或启用演示模式');
        return;
      }
      
      this.selectedCameraId = cid;
      this.showVideoModal = true;

      // Demo: 使用稳定的视频文件播放通道
      camera.demo_video_url = `/demo/video/${cid}`;

      // 兼容无法在浏览器中直接播放的格式（如 .avi）：提供 MJPEG 回退
      camera.stream_url = `/stream/${cid}`;

      // Demo: 订阅低频 SSE 结果流（检测框 + 传感器）
      this.startDemoEventSource(cid);
    },
    
    // 关闭视频弹窗
    closeVideoModal() {
      this.stopDemoEventSource();
      
      this.showVideoModal = false;
      this.selectedCameraId = null;
    },
    
    // 显示传感器详情
    showSensorDetail(sensorId) {
      const sensor = this.sensors.find(s => s.id === sensorId);
      const facility = this.facilities.find(f => f.id === sensorId);
      
      if (sensor) {
        this.selectedSensor = sensor;
        this.selectedFacility = facility;
        this.showSensorModal = true;
      }
    },
    
    // 关闭传感器模态框
    closeSensorModal() {
      this.showSensorModal = false;
      this.selectedSensor = null;
      this.selectedFacility = null;
    },
    
    // 启动定时更新
    startPeriodicUpdates() {
      // 每5秒更新系统状态
      setInterval(async () => {
        const result = await API.get('/api/system/status');
        if (result.success) {
          this.systemStatus = result.data;
        }
      }, 5000);
    }
  }
});

// 导出app实例供组件文件使用
// 注意：不在这里挂载，等组件注册完成后再挂载
