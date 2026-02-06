# 火灾检测视频数据集搜索

## 🔍 搜索关键词
- fire detection video dataset
- smoke detection temporal dataset
- fire smoke video classification
- wildfire detection dataset
- LSTM fire detection dataset

---

## 📊 推荐数据集

### 1. ⭐ Fire Detection from CCTV (Kaggle)
**链接**: https://www.kaggle.com/datasets/ritupande/fire-detection-from-cctv

**特点**:
- ✅ 视频数据集
- ✅ 包含火灾和非火灾场景
- ✅ CCTV监控视频
- ✅ 适合时序分析

**规模**:
- 火灾视频: ~500个
- 非火灾视频: ~500个
- 格式: MP4/AVI

**适用性**: ⭐⭐⭐⭐⭐ 非常适合LSTM训练

---

### 2. ⭐ Fire and Smoke Dataset (Mendeley Data)
**链接**: https://data.mendeley.com/datasets/gjxz5w7xp7/1

**特点**:
- ✅ 包含火焰和烟雾视频
- ✅ 多场景（室内/室外）
- ✅ 高质量标注
- ✅ 学术数据集

**规模**:
- 火焰视频: 300+
- 烟雾视频: 300+
- 正常场景: 300+

**适用性**: ⭐⭐⭐⭐⭐ 完美匹配3分类需求

---

### 3. ⭐ FIRESENSE Database
**链接**: http://signal.ee.bilkent.edu.tr/VisiFire/Demo/FireClips/

**特点**:
- ✅ 公开可用
- ✅ 多种火灾场景
- ✅ 包含烟雾和火焰
- ✅ 研究常用数据集

**规模**:
- 视频片段: 100+
- 场景多样性高

**适用性**: ⭐⭐⭐⭐ 适合补充训练

---

### 4. YouTube Fire Dataset (自建)
**链接**: YouTube搜索

**搜索关键词**:
- "fire detection test video"
- "smoke detection cctv"
- "fire alarm test"
- "cooking smoke vs fire smoke"
- "false fire alarm"

**特点**:
- ✅ 免费获取
- ✅ 场景真实
- ✅ 可自定义标注
- ⚠️ 需要手动下载和标注

**工具**:
- youtube-dl / yt-dlp (下载工具)
- FFmpeg (视频处理)

**适用性**: ⭐⭐⭐⭐ 适合补充特定场景

---

### 5. ⭐ Wildfire Smoke Dataset
**链接**: https://github.com/aiformankind/wildfire-smoke-dataset

**特点**:
- ✅ 野外火灾烟雾
- ✅ 时序数据
- ✅ GitHub开源
- ✅ 持续更新

**规模**:
- 图像: 10,000+
- 视频: 部分可用

**适用性**: ⭐⭐⭐ 适合烟雾检测

---

### 6. MIVIA Fire Detection Dataset
**链接**: https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/

**特点**:
- ✅ 学术标准数据集
- ✅ 视频序列
- ✅ 多场景
- ✅ 公开可用

**规模**:
- 火灾视频: 31个
- 非火灾视频: 23个

**适用性**: ⭐⭐⭐⭐ 适合基准测试

---

### 7. ⭐ Fire-Smoke-Dataset (GitHub)
**链接**: https://github.com/DeepQuestAI/Fire-Smoke-Dataset

**特点**:
- ✅ 开源数据集
- ✅ 包含火焰和烟雾
- ✅ 视频+图像
- ✅ 易于下载

**规模**:
- 多种格式
- 持续更新

**适用性**: ⭐⭐⭐⭐ 适合混合训练

---

## 🎯 推荐方案

### 方案A: 快速开始（最小数据集）
```
数据来源:
1. MIVIA Fire Detection Dataset (31+23视频)
2. FIRESENSE Database (100+视频片段)

总计: ~150个视频
标注: 手动标注为3类（无火/烟雾/火焰）

优点: 快速获取，质量高
缺点: 数据量较小
```

### 方案B: 标准训练（推荐）⭐
```
数据来源:
1. Fire Detection from CCTV (Kaggle) - 1000视频
2. Fire and Smoke Dataset (Mendeley) - 900视频
3. FIRESENSE Database - 100视频

总计: ~2000个视频
标注: 按3类标注

优点: 数据量充足，场景多样
缺点: 需要下载和整理
```

### 方案C: 完整训练（最佳效果）
```
数据来源:
1. Kaggle Fire Detection - 1000视频
2. Mendeley Fire Smoke - 900视频
3. MIVIA Dataset - 54视频
4. YouTube自采集 - 500视频
5. FIRESENSE - 100视频

总计: ~2500个视频
标注: 精细3类标注 + 场景标签

优点: 数据量大，覆盖全面
缺点: 工作量大
```

---

## 📥 数据下载指南

### Kaggle数据集下载
```bash
# 安装kaggle CLI
pip install kaggle

# 配置API密钥（从Kaggle账户获取）
# 下载数据集
kaggle datasets download -d ritupande/fire-detection-from-cctv
unzip fire-detection-from-cctv.zip -d datasets/fire_videos/
```

### YouTube视频下载
```bash
# 安装yt-dlp
pip install yt-dlp

# 下载视频
yt-dlp -f "best[height<=720]" -o "datasets/fire_videos/%(title)s.%(ext)s" [VIDEO_URL]
```

### GitHub数据集下载
```bash
# 克隆仓库
git clone https://github.com/DeepQuestAI/Fire-Smoke-Dataset.git datasets/fire_smoke/
```

---

## 🏷️ 数据标注建议

### 标注标准
```
类别0 - 无火场景:
- 正常监控画面
- 无烟无火
- 可包含其他物体

类别1 - 烟雾场景:
- 明显烟雾
- 可能有少量火焰
- 炊烟也归为此类（后期可细分）

类别2 - 火焰场景:
- 明显火焰
- 可能伴随烟雾
- 火势较大
```

### 标注工具
- **手动标注**: Excel表格记录视频文件名和标签
- **半自动**: 使用YOLO预检测，人工确认
- **格式**: CSV文件
  ```csv
  video_path,label,duration,scene_type
  videos/fire_001.mp4,2,30,indoor
  videos/smoke_001.mp4,1,25,outdoor
  videos/normal_001.mp4,0,20,indoor
  ```

---

## 📊 数据集质量要求

### 视频要求
- **分辨率**: ≥480p（推荐720p）
- **帧率**: ≥15fps（推荐30fps）
- **时长**: 10-60秒
- **格式**: MP4, AVI, MOV

### 数据分布
```
训练集: 70% (~1400视频)
验证集: 15% (~300视频)
测试集: 15% (~300视频)

类别平衡:
- 无火: 40% (~800视频)
- 烟雾: 30% (~600视频)
- 火焰: 30% (~600视频)
```

---

## 🚀 快速开始

### 步骤1: 下载推荐数据集
```bash
# 创建目录
mkdir -p datasets/fire_videos/{train,val,test}

# 下载Kaggle数据集（推荐）
kaggle datasets download -d ritupande/fire-detection-from-cctv
```

### 步骤2: 整理和标注
```python
# 使用脚本整理数据
python scripts/3_prepare_lstm_data.py
```

### 步骤3: 训练模型
```bash
python scripts/4_train_lstm.py --data_dir datasets/lstm_data --epochs 50
```

---

## 📚 参考资源

### 论文
1. "Fire Detection in Video Sequences using a Novel Probabilistic Approach"
2. "Deep Learning for Fire Detection in Videos"
3. "Temporal Fire Detection using LSTM Networks"

### GitHub项目
1. https://github.com/pedbrgs/Fire-Detection
2. https://github.com/Caoyichao/STCNet
3. https://github.com/sureshkumark23/yolo-lstm_fire_detection

---

## ✅ 推荐行动

1. **立即可做**: 下载MIVIA + FIRESENSE（~150视频）
2. **本周完成**: 下载Kaggle数据集（1000视频）
3. **下周完成**: 标注数据，开始训练
4. **可选**: 补充YouTube视频（特定场景）

---

**更新时间**: 2026年2月6日  
**状态**: 待下载数据集
