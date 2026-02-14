/**
 * 平面图Canvas组件
 */
app.component('floor-plan-canvas', {
  props: {
    floorPlan: Object,
    facilities: Array,
    cameras: Array,
    sensors: Array
  },
  
  data() {
    return {
      canvas: null,
      ctx: null,
      scale: 1.0,
      offset: { x: 0, y: 0 },
      isDragging: false,
      lastMousePos: { x: 0, y: 0 },
      floorPlanImage: null,
      imageLoaded: false,
      alertAnimationTimer: null
    };
  },
  
  watch: {
    floorPlan: {
      handler(newVal) {
        console.log('平面图数据变化:', newVal);
        if (newVal && newVal.image) {
          this.loadFloorPlanImage(newVal.image);
        } else {
          this.floorPlanImage = null;
          this.imageLoaded = false;
          this.drawCanvas();
        }
      },
      immediate: true
    },
    // 监听传感器数据变化
    sensors: {
      handler() {
        this.drawCanvas();
      },
      deep: true
    },
    // 监听摄像头数据变化
    cameras: {
      handler() {
        this.drawCanvas();
      },
      deep: true
    }
  },
  
  mounted() {
    this.initCanvas();
    // 不再使用持续动画，改为按需重绘
  },
  
  beforeUnmount() {
    // 清理事件监听
    window.removeEventListener('resize', this.resizeCanvas);
  },
  
  methods: {
    getThemeColor(varName, fallback) {
      try {
        const value = getComputedStyle(document.documentElement)
          .getPropertyValue(varName)
          .trim();
        return value || fallback;
      } catch (e) {
        return fallback;
      }
    },

    initCanvas() {
      this.canvas = this.$refs.canvas;
      if (!this.canvas) return;
      
      this.ctx = this.canvas.getContext('2d');
      this.resizeCanvas();
      
      window.addEventListener('resize', this.resizeCanvas);
    },
    
    loadFloorPlanImage(imagePath) {
      console.log('加载平面图:', imagePath);
      const img = new Image();
      img.onload = () => {
        this.floorPlanImage = img;
        this.imageLoaded = true;
        console.log('平面图加载成功');
        
        // 自动调整缩放和位置，使平面图居中并适应屏幕
        this.fitToScreen();
        
        this.drawCanvas();
      };
      img.onerror = (e) => {
        console.error('平面图加载失败:', imagePath, e);
        this.floorPlanImage = null;
        this.imageLoaded = false;
        this.drawCanvas();
      };
      img.src = imagePath;
    },
    
    fitToScreen() {
      if (!this.floorPlanImage || !this.canvas) return;
      
      const imgWidth = this.floorPlan.width || this.floorPlanImage.width;
      const imgHeight = this.floorPlan.height || this.floorPlanImage.height;
      
      const canvasWidth = this.canvas.width;
      const canvasHeight = this.canvas.height;
      
      // 计算缩放比例，留出10%的边距
      const scaleX = (canvasWidth * 0.9) / imgWidth;
      const scaleY = (canvasHeight * 0.9) / imgHeight;
      this.scale = Math.min(scaleX, scaleY, 1.0); // 不放大，只缩小
      
      // 计算居中偏移
      this.offset.x = (canvasWidth - imgWidth * this.scale) / 2;
      this.offset.y = (canvasHeight - imgHeight * this.scale) / 2;
    },
    
    resizeCanvas() {
      const container = this.canvas.parentElement;
      this.canvas.width = container.clientWidth;
      this.canvas.height = container.clientHeight;
      this.drawCanvas();
    },
    
    drawCanvas() {
      if (!this.ctx) return;
      
      const ctx = this.ctx;
      const canvas = this.canvas;
      
      // 清空画布
      ctx.fillStyle = this.getThemeColor('--color-bg-primary', '#1C1C1E');
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // 保存状态
      ctx.save();
      
      // 应用缩放和平移
      ctx.translate(this.offset.x, this.offset.y);
      ctx.scale(this.scale, this.scale);
      
      // 绘制平面图
      this.drawFloorPlan(ctx);
      
      // 绘制设施图标
      this.drawFacilities(ctx);
      
      // 恢复状态
      ctx.restore();
    },
    
    drawFloorPlan(ctx) {
      if (!this.floorPlan || !this.floorPlan.image) {
        // 显示占位文本
        ctx.fillStyle = '#8E8E93';
        ctx.font = '24px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          '平面图未配置', 
          this.canvas.width / 2 / this.scale - this.offset.x / this.scale, 
          this.canvas.height / 2 / this.scale - this.offset.y / this.scale
        );
        return;
      }
      
      // 绘制平面图图片
      if (this.imageLoaded && this.floorPlanImage) {
        const imgWidth = this.floorPlan.width || this.floorPlanImage.width;
        const imgHeight = this.floorPlan.height || this.floorPlanImage.height;
        
        // 绘制在原点，缩放和偏移由 canvas transform 处理
        ctx.drawImage(this.floorPlanImage, 0, 0, imgWidth, imgHeight);
      } else {
        // 加载中
        ctx.fillStyle = '#8E8E93';
        ctx.font = '18px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          '平面图加载中...', 
          this.canvas.width / 2 / this.scale - this.offset.x / this.scale, 
          this.canvas.height / 2 / this.scale - this.offset.y / this.scale
        );
      }
    },
    
    drawFacilities(ctx) {
      // 分别统计摄像头和传感器的序号
      let cameraIndex = 0;
      let sensorIndex = 0;
      
      this.facilities.forEach(facility => {
        let displayIndex = null;
        
        if (facility.type === 'camera') {
          displayIndex = ++cameraIndex;
        } else if (facility.type === 'temperature_sensor' || facility.type === 'humidity_sensor' || facility.type === 'smoke_detector') {
          displayIndex = ++sensorIndex;
        }
        
        this.drawFacilityIcon(ctx, facility, displayIndex);
      });
    },
    
    drawFacilityIcon(ctx, facility, displayIndex) {
      const x = facility.position.x;
      const y = facility.position.y;
      const size = 30;
      
      // 获取图标和颜色
      const { icon, color } = this.getFacilityStyle(facility);
      
      // 绘制背景圆圈
      ctx.save();
      ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
      ctx.shadowBlur = 10;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 6;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, size / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      
      // 绘制图标
      ctx.font = `${size * 0.6}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      // 描边提升可读性
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)';
      ctx.strokeText(icon, x, y);
      ctx.fillText(icon, x, y);
      
      // 如果有序号，智能选择显示位置
      if (displayIndex !== null) {
        const imgWidth = this.floorPlan?.width || this.floorPlanImage?.width || this.canvas.width;
        const imgHeight = this.floorPlan?.height || this.floorPlanImage?.height || this.canvas.height;
        const margin = 25; // 边距阈值
        
        // 计算序号文本的大小
        ctx.font = 'bold 14px sans-serif';
        const textWidth = ctx.measureText(displayIndex.toString()).width;
        
        // 默认位置：上方
        let textX = x;
        let textY = y - size / 2 - 10;
        
        // 检测是否靠近上边缘
        if (y < margin) {
          // 改为下方
          textY = y + size / 2 + 18;
        }
        
        // 检测是否靠近左边缘
        if (x < margin + textWidth / 2) {
          // 改为右侧
          textX = x + size / 2 + 15;
          textY = y;
        }
        
        // 检测是否靠近右边缘
        if (x > imgWidth - margin - textWidth / 2) {
          // 改为左侧
          textX = x - size / 2 - 15;
          textY = y;
        }
        
        // 绘制序号（带描边以提高可读性）
        ctx.fillStyle = '#FFFFFF';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.strokeText(displayIndex.toString(), textX, textY);
        ctx.fillText(displayIndex.toString(), textX, textY);
      }
      
      // 如果是告警状态，添加闪烁效果（使用定时器而不是持续动画）
      const sensor = this.sensors.find(s => s.id === facility.id);
      const camera = this.cameras.find(c => c.id === facility.id);
      
      if ((camera && camera.alert_status) || (sensor && sensor.status === 'alert')) {
        const time = Date.now() / 1000;
        const alpha = 0.5 + 0.5 * Math.sin(time * 3);
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, size / 2 + 5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1.0;
        
        // 为告警状态设置定时重绘（实现闪烁效果）
        if (!this.alertAnimationTimer) {
          this.alertAnimationTimer = setInterval(() => {
            this.drawCanvas();
          }, 100); // 每100ms重绘一次，实现闪烁
        }
      } else {
        // 没有告警时清除定时器
        if (this.alertAnimationTimer) {
          clearInterval(this.alertAnimationTimer);
          this.alertAnimationTimer = null;
        }
      }
    },
    
    getFacilityStyle(facility) {
      let icon = '📍';
      let color = '#00D9FF';
      
      switch (facility.type) {
        case 'camera':
          icon = '📹';
          const camera = this.cameras.find(c => c.id === facility.id);
          if (camera) {
            if (camera.alert_status === 'fire') {
              color = '#FF3B30';
            } else if (camera.alert_status === 'smoke') {
              color = '#FF9500';
            } else if (camera.status === 'online') {
              color = '#34C759';
            } else {
              color = '#8E8E93';
            }
          }
          break;
        case 'sprinkler':
          icon = '💧';
          color = '#5AC8FA';
          break;
        case 'temperature_sensor':
          icon = '🌡️';
          const tempSensor = this.sensors.find(s => s.id === facility.id);
          if (tempSensor && tempSensor.status === 'alert') {
            color = '#FF3B30';
          } else {
            color = '#34C759';
          }
          break;
        case 'humidity_sensor':
          icon = '💨';
          color = '#5AC8FA';
          break;
        case 'smoke_detector':
          icon = '☁️';
          const smokeSensor = this.sensors.find(s => s.id === facility.id);
          if (smokeSensor && smokeSensor.status === 'alert') {
            color = '#FF3B30'; // 告警：红色
          } else {
            color = '#34C759'; // 正常：绿色
          }
          break;
        case 'fire_extinguisher':
          icon = '🧯';
          color = '#FF453A';
          break;
        case 'obstacle':
          icon = '🚧';
          color = '#FFD60A';
          break;
      }
      
      return { icon, color };
    },
    
    handleMouseDown(e) {
      this.isDragging = true;
      this.lastMousePos = { x: e.clientX, y: e.clientY };
      this.canvas.style.cursor = 'grabbing';
    },
    
    handleMouseMove(e) {
      if (this.isDragging) {
        const dx = e.clientX - this.lastMousePos.x;
        const dy = e.clientY - this.lastMousePos.y;
        this.offset.x += dx;
        this.offset.y += dy;
        this.lastMousePos = { x: e.clientX, y: e.clientY };
        this.drawCanvas();
      }
    },
    
    handleMouseUp() {
      this.isDragging = false;
      this.canvas.style.cursor = 'grab';
    },
    
    handleWheel(e) {
      e.preventDefault();
      
      // 获取鼠标在 canvas 上的位置
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      
      // 计算鼠标在世界坐标系中的位置（缩放前）
      const worldX = (mouseX - this.offset.x) / this.scale;
      const worldY = (mouseY - this.offset.y) / this.scale;
      
      // 计算新的缩放比例
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newScale = this.scale * delta;
      
      // 限制缩放范围
      if (newScale >= 0.1 && newScale <= 5.0) {
        // 更新缩放
        this.scale = newScale;
        
        // 调整偏移，使鼠标位置保持不变
        this.offset.x = mouseX - worldX * this.scale;
        this.offset.y = mouseY - worldY * this.scale;
        
        this.drawCanvas();
      }
    },
    
    handleClick(e) {
      const rect = this.canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left - this.offset.x) / this.scale;
      const y = (e.clientY - rect.top - this.offset.y) / this.scale;
      
      // 检查是否点击了设施
      for (const facility of this.facilities) {
        const dx = x - facility.position.x;
        const dy = y - facility.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 20) {
          this.$emit('facility-click', facility);
          break;
        }
      }
    },
    
    zoomIn() {
      // 以画布中心为基准缩放
      const centerX = this.canvas.width / 2;
      const centerY = this.canvas.height / 2;
      
      const worldX = (centerX - this.offset.x) / this.scale;
      const worldY = (centerY - this.offset.y) / this.scale;
      
      this.scale *= 1.2;
      if (this.scale > 5.0) this.scale = 5.0;
      
      this.offset.x = centerX - worldX * this.scale;
      this.offset.y = centerY - worldY * this.scale;
      
      this.drawCanvas();
    },
    
    zoomOut() {
      // 以画布中心为基准缩放
      const centerX = this.canvas.width / 2;
      const centerY = this.canvas.height / 2;
      
      const worldX = (centerX - this.offset.x) / this.scale;
      const worldY = (centerY - this.offset.y) / this.scale;
      
      this.scale *= 0.8;
      if (this.scale < 0.1) this.scale = 0.1;
      
      this.offset.x = centerX - worldX * this.scale;
      this.offset.y = centerY - worldY * this.scale;
      
      this.drawCanvas();
    },
    
    resetView() {
      // 重置为适应屏幕的状态
      this.fitToScreen();
      this.drawCanvas();
    }
  },
  
  template: `
    <div class="floor-plan-container">
      <canvas 
        ref="canvas"
        class="floor-plan-canvas"
        @mousedown="handleMouseDown"
        @mousemove="handleMouseMove"
        @mouseup="handleMouseUp"
        @wheel="handleWheel"
        @click="handleClick"
      ></canvas>
      
      <div class="floor-plan-controls">
        <button class="control-button" @click="zoomIn" title="放大">+</button>
        <button class="control-button" @click="zoomOut" title="缩小">−</button>
        <button class="control-button" @click="resetView" title="重置">⟲</button>
      </div>
    </div>
  `
});
