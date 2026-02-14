# 建筑配置模板

复制此文件夹创建新建筑：

1. 复制 `.template` 文件夹，重命名为你的建筑名称
2. 编辑 `config.json` 填写建筑信息
3. 添加平面图到 `floor_plans/` 目录
4. 在 `config/buildings.json` 中添加索引

## 配置说明

- `id`: 建筑唯一标识
- `name`: 建筑名称
- `floors`: 楼层列表
  - `facilities`: 设施列表（摄像头、传感器等）

## 设施类型

- `camera` - 摄像头
- `temperature_sensor` - 温度传感器
- `humidity_sensor` - 湿度传感器
- `smoke_detector` - 烟雾探测器
- `sprinkler` - 喷水器
- `fire_extinguisher` - 灭火器
