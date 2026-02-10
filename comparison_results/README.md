## 📊 生成的图表说明

### 1. accuracy_comparison.png
- **内容**：准确率对比柱状图
- **显示**：YOLO 50% vs LSTM 91%
- **用途**：直观看出LSTM更准确

### 2. confusion_matrix.png
- **内容**：混淆矩阵（预测 vs 真实）
- **左图**：YOLO的预测分布
- **右图**：LSTM的预测分布
- **用途**：看出哪些类别容易判断错

### 3. detection_stats.png
- **内容**：4个子图的详细统计
  - 各类别平均置信度
  - 检测帧数分布
  - 各类别准确率
  - 处理速度对比
- **用途**：全面对比两种方法

### 4. lstm_advantages.png
- **内容**：LSTM的时序优势
  - 稳定性对比（波动性）
  - 连续性检测能力
  - 趋势识别能力
  - 判断逻辑对比
- **用途**：展示LSTM为什么更强

---

## 📁 JSON文件结构说明

### 文件位置
`comparison_results/comparison_report.json`

### 文件结构
```json
{
  "timestamp": "测试时间",
  "summary": { 总结数据 },
  "details": [ 每个视频的详细结果 ]
}
```

---

## 1️⃣ Summary（总结部分）- 整体成绩单

```json
"summary": {
  "total_videos": 34,        // 一共测试了34个视频
  "yolo_accuracy": 50.0,     // YOLO准确率：50分（满分100）
  "lstm_accuracy": 91.18,    // LSTM准确率：91分（满分100）
  "yolo_correct": 17,        // YOLO答对了17个
  "lstm_correct": 31         // LSTM答对了31个
}
```

---

## 2️⃣ Details（详细结果）- 每个视频的分析

每个视频包含以下信息：

### 基本信息
```json
{
  "video_name": "mivia_fire_fire2.avi",     // 视频文件名
  "video_path": "datasets\\...",            // 视频完整路径
  "ground_truth": "fire",                   // 正确答案：这是火焰视频
  "yolo": { YOLO的答题过程 },
  "lstm": { LSTM的答题过程 }
}
```

---

## 3️⃣ YOLO字段详解

```json
"yolo": {
  "method": "YOLO",                    // 方法名称
  "total_frames": 140,                 // 视频总共140帧（140张图片）
  "fps": 29.97,                        // 每秒29.97帧
  
  // ===== 检测统计 =====
  "detections": {
    "fire": 6,      // 有6帧检测到了火焰
    "smoke": 0,     // 有0帧检测到了烟雾
    "none": 134     // 有134帧什么都没检测到
  },
  
  // ===== 检测比例 =====
  "detection_ratios": {
    "fire": 0.043,    // 火焰占4.3% (6÷140)
    "smoke": 0.0,     // 烟雾占0%
    "none": 0.957     // 无检测占95.7%
  },
  
  // ===== 性能指标 =====
  "avg_confidence": 0.356,           // 平均置信度：35.6%（检测的把握程度）
  "volatility": 0.057,               // 波动性：5.7%（检测结果变化频率，越低越稳定）
  
  // ===== 最终判断 =====
  "prediction": "normal",            // 最终判断：正常（无火灾）
  "confidence_level": 0.957,         // 判断置信度：95.7%
  
  // ===== 是否检测到 =====
  "has_fire": true,                  // 是否检测到过火焰：是
  "has_smoke": false                 // 是否检测到过烟雾：否
}
```

### YOLO判断逻辑
```
1. 看140张图片
2. 只在6张图片里看到火焰（4.3%）
3. 因为比例太低（<5%），判断为"正常"
4. 结果：错了！正确答案是"fire"
```

**问题**：
- 只看单张图片，不考虑前后关系
- 4.3%的火焰可能是真火灾的早期阶段

---

## 4️⃣ LSTM字段详解

```json
"lstm": {
  "method": "YOLO+LSTM",
  "total_frames": 140,
  "fps": 29.97,
  
  // ===== YOLO检测结果（和上面一样）=====
  "yolo_detections": {
    "fire": 6,
    "smoke": 0,
    "none": 134
  },
  
  // ===== LSTM预测结果（分析30帧序列后的判断）=====
  "lstm_predictions": {
    "0": 21,     // 预测为"无火"的次数：21次
    "1": 1,      // 预测为"烟雾"的次数：1次
    "2": 89      // 预测为"火焰"的次数：89次
  },
  
  // ===== LSTM预测比例 =====
  "lstm_ratios": {
    "fire": 0.802,      // 火焰占80.2%
    "smoke": 0.009,     // 烟雾占0.9%
    "normal": 0.189     // 无火占18.9%
  },
  
  // ===== 性能指标 =====
  "avg_confidence": 0.781,           // 平均置信度：78.1%（比YOLO高！）
  "volatility": 0.036,               // 波动性：3.6%（比YOLO低！更稳定）
  
  // ===== LSTM独有的时序特征 =====
  "max_continuous_fire": 51,         // 最长连续火焰帧：51帧
  "max_continuous_smoke": 1,         // 最长连续烟雾帧：1帧
  "smoke_to_fire_transitions": 1,    // 烟雾→火焰转变次数：1次
  
  // ===== 最终判断 =====
  "prediction": "fire",              // 最终判断：火焰
  "confidence_level": 0.802,         // 判断置信度：80.2%
  
  "has_fire": true,
  "has_smoke": true
}
```

### LSTM判断逻辑
```
1. 看同样的140张图片
2. 但会"记忆"前面看到的内容
3. 发现：连续51帧都是火焰=
4. 发现：有烟雾→火焰的发展过程
5. 判断：这是真的火灾！
```

**优势**：
- 分析视频的时序变化
- 识别持续性特征（连续51帧）
- 捕捉发展趋势（烟雾→火焰）
- 更准确可靠

---

## 💡 使用说明

### 查看测试结果
```bash
# 查看JSON详细结果
notepad comparison_results/comparison_report.json

# 查看图表（Windows）
start comparison_results/accuracy_comparison.png
start comparison_results/confusion_matrix.png
start comparison_results/detection_stats.png
start comparison_results/lstm_advantages.png
```