# EmberGuard AI - 脚本工具

本目录包含项目的所有训练和运行脚本，按执行顺序编号。

## 📋 脚本列表

### 0️⃣ 0_download_datasets.py
**功能**: 数据集下载辅助工具

**使用**:
```bash
python scripts/0_download_datasets.py
```

**说明**:
- 创建必要的目录结构
- 显示所有可用数据集的下载链接
- 提供下载和整理指南
- 创建标注文件模板

**输出**:
- 创建 `datasets/fire_videos/` 目录结构
- 创建 `annotations_template.csv` 模板

---

### 1️⃣ 1_train_yolo.py
**功能**: 训练YOLOv8火灾检测模型

**使用**:
```bash
python scripts/1_train_yolo.py
```

**说明**:
- 使用D-Fire数据集训练
- 配置文件: `configs/yolo_fire.yaml`
- 输出目录: `runs/detect/train*/`
- 训练完成后会生成最佳模型 `best.pt`

---

### 2️⃣ 2_validate_yolo.py
**功能**: 验证YOLOv8模型性能

**使用**:
```bash
python scripts/2_validate_yolo.py
```

**说明**:
- 在验证集上评估模型
- 输出精度、召回率、mAP等指标
- 生成混淆矩阵和PR曲线

---

### 3️⃣ 3_prepare_lstm_data.py
**功能**: 准备LSTM训练数据

**使用**:
```python
from scripts.prepare_lstm_data import LSTMDataPreparer

preparer = LSTMDataPreparer(
    yolo_model_path='runs/detect/train2/weights/best.pt',
    sequence_length=30
)

# 准备数据集
video_list = [
    ('videos/no_fire_1.mp4', 0),  # 无火
    ('videos/smoke_1.mp4', 1),     # 烟雾
    ('videos/fire_1.mp4', 2),      # 火焰
]

preparer.prepare_dataset(video_list, 'datasets/lstm_data')
```

**说明**:
- 从视频中提取YOLO特征序列
- 生成训练数据: `sequences.npy`, `labels.npy`
- 需要准备标注好的视频数据

---

### 4️⃣ 4_train_lstm.py
**功能**: 训练LSTM时序分类模型

**使用**:
```bash
python scripts/4_train_lstm.py \
    --data_dir datasets/lstm_data \
    --output_dir models/lstm \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

**参数**:
- `--data_dir`: 数据目录（由步骤3生成）
- `--output_dir`: 模型输出目录
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--hidden_size`: LSTM隐藏层大小（默认128）
- `--num_layers`: LSTM层数（默认2）
- `--dropout`: Dropout比例（默认0.3）

**输出**:
- `best.pt`: 最佳模型
- `last.pt`: 最终模型
- `history.json`: 训练历史
- `config.json`: 训练配置

---

### 5️⃣ 5_run_gui.py
**功能**: 启动图形化检测界面

**使用**:
```bash
python scripts/5_run_gui.py
```

**功能**:
- 图片/视频/摄像头检测
- 实时显示检测结果
- 保存检测结果
- 支持YOLO或YOLO+LSTM模式

---

## 🔄 完整工作流程

### Phase 0: 数据准备
```bash
# 0. 下载数据集（辅助工具）
python scripts/0_download_datasets.py
```

### Phase 1: YOLO训练
```bash
# 1. 训练YOLO模型
python scripts/1_train_yolo.py

# 2. 验证模型性能
python scripts/2_validate_yolo.py
```

### Phase 2: LSTM训练
```bash
# 3. 准备LSTM数据（需要视频数据）
python scripts/3_prepare_lstm_data.py

# 4. 训练LSTM模型
python scripts/4_train_lstm.py --data_dir datasets/lstm_data --epochs 50
```

### Phase 3: 使用
```bash
# 5. 启动GUI进行检测
python scripts/5_run_gui.py
```

---

## 📊 数据要求

### YOLO训练数据
- 格式: YOLO格式（图片+标注txt）
- 位置: `datasets/D-Fire/`
- 结构:
  ```
  D-Fire/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/
  ```

### LSTM训练数据
- 格式: 视频文件（mp4, avi等）
- 标注: 每个视频对应一个标签
  - 0: 无火场景
  - 1: 烟雾场景
  - 2: 火焰场景
- 建议: 每类至少10个视频，每个视频30秒以上

---

## 🎯 性能目标

| 模型 | 准确率 | 误报率 | 推理速度 |
|------|--------|--------|----------|
| YOLO | >95% | - | ~30ms |
| LSTM | >99% | <2% | ~10ms |
| 组合 | >99% | <2% | ~40ms |

---

## 💡 使用建议

1. **首次使用**: 按顺序执行脚本1→2→5，先体验基础YOLO检测
2. **进阶使用**: 准备视频数据后执行3→4，训练LSTM模型
3. **生产部署**: 使用YOLO+LSTM组合模式，获得最佳性能

---

## 🔗 相关文档

- 项目文档: `../README.md`
- 开发日志: `../DEVELOPMENT_LOG.md`
- LSTM模块: `../emberguard/README.md`
- 技术研究: `../docs/TECHNICAL_RESEARCH.md`
## 2026-03 训练脚本更新

- `scripts/4_train_lstm.py` 现在默认启用轻量 `Temporal Attention Pooling`。
- 该机制用于替代旧版“只使用最后一个时间步输出”的做法，适合小样本下提升对中间帧短暂火焰/烟雾特征的关注。
- 如需做消融或回退旧结构，请添加参数：`--disable_temporal_attention`。
- 训练产生的 `config.json` 会额外保存 `use_temporal_attention` 配置，便于复现实验。
