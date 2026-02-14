# 建筑数据目录

每个建筑一个独立文件夹，包含：
- `config.json` - 建筑配置（楼层、设施）
- `alerts/` - 告警记录
- `history/` - 历史数据
- `videos/` - 录像文件
- `floor_plans/` - 平面图

## 目录结构示例

```
buildings/
├── building_001/           # 建筑1
│   ├── config.json        # 建筑配置
│   ├── alerts/            # 告警记录
│   ├── history/           # 历史数据
│   ├── videos/            # 录像文件
│   └── floor_plans/       # 平面图
│       ├── floor_1.png
│       └── floor_2.png
├── building_002/           # 建筑2
│   └── ...
└── demo/                   # 演示建筑
    └── ...
```

## 添加新建筑

1. 创建建筑文件夹：`buildings/your_building_name/`
2. 创建配置文件：`config.json`
3. 在主配置 `config/buildings.json` 中添加索引
