# 配置文件说明

## system_config.json - 系统配置

系统的主配置文件，包含全局设置和设施配置。

### 主要字段

- `demo_mode`: 演示模式开关（true/false）
- `floor_plan`: 平面图配置
  - `image`: 平面图图片路径
  - `width`: 平面图宽度
  - `height`: 平面图高度
- `facilities`: 设施列表数组
- `models`: AI模型路径配置
  - `yolo`: YOLO模型路径
  - `lstm`: LSTM模型路径
- `detection`: 检测参数配置
  - `normal_fps`: 正常检测帧率
  - `alert_fps`: 告警检测帧率
  - `alert_cooldown`: 告警冷却时间（秒）

### 设施类型

#### 摄像头 (camera)
```json
{
  "id": "cam_001",
  "type": "camera",
  "name": "东侧大殿",
  "position": { "x": 300, "y": 200 },
  "source": 0,
  "demo_video": "/static/videos/demo_fire.mp4",
  "enabled": true,
  "detection_config": {
    "conf_threshold": 0.25,
    "normal_fps": 5,
    "alert_fps": 15
  }
}
```

#### 温度传感器 (temperature_sensor)
```json
{
  "id": "temp_sensor_001",
  "type": "temperature_sensor",
  "name": "温度传感器1",
  "position": { "x": 350, "y": 220 },
  "threshold": 60,
  "unit": "°C",
  "enabled": true
}
```

#### 湿度传感器 (humidity_sensor)
```json
{
  "id": "humidity_sensor_001",
  "type": "humidity_sensor",
  "name": "湿度传感器1",
  "position": { "x": 380, "y": 240 },
  "threshold": 80,
  "unit": "%",
  "enabled": true
}
```

#### 其他设施类型
- `sprinkler` - 喷水器
- `smoke_detector` - 烟感器
- `fire_extinguisher` - 灭火器
- `obstacle` - 遮挡物

## buildings.json - 建筑配置

多建筑/楼层配置文件，支持切换不同建筑和楼层。

### 结构

```json
{
  "buildings": [
    {
      "id": "main_hall",
      "name": "主殿",
      "floors": [
        {
          "id": "floor_1",
          "name": "一层",
          "floor_plan": { ... },
          "facilities": [ ... ]
        }
      ]
    }
  ]
}
```

### 使用场景

1. **单层建筑**: 直接在building下配置floor_plan和facilities
2. **多层建筑**: 使用floors数组，每层独立配置
3. **多建筑**: buildings数组中添加多个建筑

## 配置修改

### 添加新摄像头

1. 在`facilities`数组中添加新的摄像头配置
2. 设置唯一的`id`
3. 配置`position`坐标（相对于平面图）
4. 设置`source`（真实摄像头）和`demo_video`（演示视频）

### 切换演示模式

修改`demo_mode`字段：
- `true`: 使用演示视频
- `false`: 使用真实摄像头

### 调整检测参数

修改`detection`配置：
- `normal_fps`: 降低可节省资源
- `alert_fps`: 提高可获得更快响应
- `alert_cooldown`: 调整告警冷却时间

## 注意事项

1. 修改配置后需要重启后端服务
2. 确保所有文件路径正确
3. 坐标系原点在左上角
4. ID必须唯一
5. 配置文件使用UTF-8编码
