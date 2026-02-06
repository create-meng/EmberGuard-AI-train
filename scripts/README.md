# 脚本说明

## 可用脚本

### 1. run_gui.py - 运行GUI界面
启动图形化界面进行火灾检测

```bash
python scripts/run_gui.py
```

**功能：**
- 图片/视频检测
- 实时摄像头检测
- 可视化结果展示
- 参数调节

### 2. train_model.py - 训练模型
训练YOLO火灾检测模型

```bash
python scripts/train_model.py
```

**配置参数：**
- 模型: `models/yolov8n.pt`
- 数据配置: `configs/yolo_fire.yaml`
- Epochs: 50
- Batch size: 48

### 3. validate_model.py - 验证模型
在验证集上评估模型性能

```bash
python scripts/validate_model.py
```

**输出指标：**
- mAP50
- mAP50-95
- Precision
- Recall

### 4. test_model.py - 测试模型
测试模型在不同输入源上的表现

```bash
# 测试图片
python scripts/test_model.py --source image.jpg

# 测试视频
python scripts/test_model.py --source video.mp4

# 测试摄像头
python scripts/test_model.py --source 0

# 测试文件夹
python scripts/test_model.py --source path/to/images/

# 使用自定义模型和置信度
python scripts/test_model.py --source 0 --model runs/detect/train2/weights/best.pt --conf 0.5
```

**参数说明：**
- `--source`: 输入源（图片/视频路径、摄像头编号、文件夹）
- `--model`: 模型路径（默认: runs/detect/train2/weights/best.pt）
- `--conf`: 置信度阈值（默认: 0.25）
- `--no-save`: 不保存检测结果

## 快速开始

### 新手推荐
使用GUI界面，最简单直观：
```bash
python scripts/run_gui.py
```

### 命令行用户
快速测试摄像头：
```bash
python scripts/test_model.py --source 0
```

### 开发者
训练自己的模型：
```bash
python scripts/train_model.py
python scripts/validate_model.py
```
