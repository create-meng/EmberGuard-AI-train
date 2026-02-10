## 生成的图表

### 1. accuracy_comparison.png
- 准确率对比柱状图
- 直观展示YOLO vs YOLO+LSTM的准确率差异

### 2. confusion_matrix.png
- 混淆矩阵对比
- 左图：YOLO的预测分布
- 右图：YOLO+LSTM的预测分布
- 可以看出LSTM在各类别上的预测更准确

### 3. detection_stats.png
- 检测统计详细对比（4个子图）
  - 各类别平均置信度
  - 检测帧数分布
  - 各类别准确率
  - 处理速度对比

### 4. lstm_advantages.png
- **LSTM时序优势对比**（4个子图）
  - **稳定性对比**: LSTM波动性更低，检测更稳定
  - **连续性检测**: LSTM能识别持续性火灾特征
  - **趋势识别**: LSTM能捕捉烟雾→火焰的发展过程
  - **判断逻辑对比**: 详细说明两种方法的差异

## 核心发现

### YOLO的问题
1. **逐帧独立判断**: 每帧单独分析，无时序信息
2. **波动性大**: 容易受单帧误检影响
3. **无法识别趋势**: 不能判断火灾发展过程
4. **准确率低**: 仅50%，不适合实际应用

### LSTM的优势
1. **时序连续性**: 分析30帧序列，不是单帧判断
2. **稳定性高**: 时序平滑，减少波动和误报
3. **趋势识别**: 能捕捉烟雾→火焰的发展过程
4. **准确率高**: 91.2%，适合实际部署

## 判断逻辑对比

### YOLO判断逻辑
```
1. 逐帧检测火焰/烟雾
2. 统计检测比例
3. 如果 fire_ratio > 5%: 判断为火焰
4. 如果 smoke_ratio > 5%: 判断为烟雾
5. 否则: 判断为正常
```

**问题**: 
- 单帧误检会影响结果
- 无法判断是否是持续性火灾
- 不能识别火灾发展趋势

### YOLO+LSTM判断逻辑
```
1. YOLO提取每帧特征（8维）
2. LSTM分析30帧序列
3. 考虑连续性（>10帧）
4. 识别趋势（烟雾→火焰）
5. 时序平滑（最近10次预测平均）
6. 综合判断
```

**优势**:
- 时序分析，不受单帧影响
- 识别持续性火灾特征
- 捕捉火灾发展过程
- 更稳定可靠

## 使用说明

### 查看测试结果
```bash
# 查看详细JSON结果
cat comparison_results/comparison_report.json

# 查看图表
start comparison_results/accuracy_comparison.png
start comparison_results/confusion_matrix.png
start comparison_results/detection_stats.png
start comparison_results/lstm_advantages.png
```