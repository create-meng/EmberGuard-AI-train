"""
后处理分析器 - 分析检测序列的趋势和模式
"""
import numpy as np
from collections import deque


class PostProcessor:
    """
    后处理分析器
    
    分析检测序列，提取趋势特征：
    - 火焰/烟雾的持续性
    - 检测区域的变化趋势
    - 置信度变化曲线
    - 判断是否是真实火灾
    """
    
    def __init__(self, window_size=30):
        """
        初始化
        
        Args:
            window_size: 分析窗口大小（帧数）
        """
        self.window_size = window_size
        self.detection_history = deque(maxlen=1000)  # 保存最近1000帧的检测历史
        
    def add_detection(self, result):
        """
        添加一帧检测结果
        
        Args:
            result: 检测结果字典
        """
        # 提取关键信息
        frame_info = {
            'has_detection': result.get('has_detection', False),
            'yolo_detections': len(result.get('yolo_detections', [])),
            'lstm_prediction': result.get('lstm_prediction', None),
            'lstm_confidence': result.get('lstm_confidence', 0.0),
            'features': result.get('features', None),
            'yolo_boxes': result.get('yolo_detections', [])  # 保存完整的检测框信息
        }
        
        self.detection_history.append(frame_info)
    
    def analyze_sequence(self, min_frames=30):
        """
        分析检测序列
        
        Args:
            min_frames: 最少需要的帧数
            
        Returns:
            dict: 分析结果
        """
        if len(self.detection_history) < min_frames:
            return {
                'status': 'insufficient_data',
                'message': f'数据不足，需要至少{min_frames}帧，当前{len(self.detection_history)}帧'
            }
        
        # 转换为numpy数组便于分析
        history = list(self.detection_history)
        
        # 1. 检测持续性分析
        detection_rate = sum(1 for h in history if h['has_detection']) / len(history)
        
        # 2. LSTM预测分布
        lstm_predictions = [h['lstm_prediction'] for h in history if h['lstm_prediction'] is not None]
        if lstm_predictions:
            pred_counts = {0: 0, 1: 0, 2: 0}  # 无火、烟雾、火焰
            for pred in lstm_predictions:
                pred_counts[pred] += 1
            
            total_preds = len(lstm_predictions)
            pred_distribution = {
                'normal': pred_counts[0] / total_preds,
                'smoke': pred_counts[1] / total_preds,
                'fire': pred_counts[2] / total_preds
            }
        else:
            pred_distribution = None
        
        # 3. 置信度趋势分析
        confidences = [h['lstm_confidence'] for h in history if h['lstm_confidence'] > 0]
        if confidences:
            conf_mean = np.mean(confidences)
            conf_std = np.std(confidences)
            conf_trend = self._calculate_trend(confidences)
        else:
            conf_mean = 0
            conf_std = 0
            conf_trend = 0
        
        # 4. 检测区域变化分析（基于特征）
        area_changes = []
        position_changes = []  # 位置变化（运动）
        
        for i in range(1, len(history)):
            if history[i]['features'] is not None and history[i-1]['features'] is not None:
                # 特征向量: [x_center, y_center, width, height, area, aspect_ratio, confidence, class_id]
                # 面积变化
                area_curr = history[i]['features'][4]
                area_prev = history[i-1]['features'][4]
                if area_prev > 0:
                    area_change = (area_curr - area_prev) / area_prev
                    area_changes.append(area_change)
                
                # 位置变化（中心点移动距离）
                x_curr, y_curr = history[i]['features'][0], history[i]['features'][1]
                x_prev, y_prev = history[i-1]['features'][0], history[i-1]['features'][1]
                position_change = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
                position_changes.append(position_change)
        
        if area_changes:
            area_expansion_rate = np.mean(area_changes)
            area_volatility = np.std(area_changes)
        else:
            area_expansion_rate = 0
            area_volatility = 0
        
        # 运动分析
        if position_changes:
            movement_rate = np.mean(position_changes)  # 平均移动速度
            movement_volatility = np.std(position_changes)  # 移动波动性
        else:
            movement_rate = 0
            movement_volatility = 0
        
        # 5. 烟雾/火焰扩散模式分析
        smoke_fire_pattern = self._analyze_smoke_fire_pattern(history, pred_distribution)
        
        # 6. 判断是否是真实火灾
        fire_score = self._calculate_fire_score(
            detection_rate,
            pred_distribution,
            conf_mean,
            area_expansion_rate,
            area_volatility,
            movement_rate,
            smoke_fire_pattern
        )
        
        # 7. 生成分析报告
        analysis = {
            'status': 'success',
            'total_frames': len(history),
            'detection_rate': detection_rate,
            'prediction_distribution': pred_distribution,
            'confidence': {
                'mean': conf_mean,
                'std': conf_std,
                'trend': conf_trend  # >0表示上升，<0表示下降
            },
            'area_analysis': {
                'expansion_rate': area_expansion_rate,  # >0表示扩散，<0表示收缩
                'volatility': area_volatility  # 波动性，火焰跳动会导致高波动
            },
            'movement_analysis': {
                'rate': movement_rate,  # 平均移动速度
                'volatility': movement_volatility  # 移动波动性
            },
            'pattern_analysis': smoke_fire_pattern,  # 烟雾/火焰扩散模式
            'fire_score': fire_score,  # 0-100，越高越可能是真实火灾
            'conclusion': self._generate_conclusion(fire_score, pred_distribution, smoke_fire_pattern)
        }
        
        return analysis
    
    def _calculate_trend(self, values, window=10):
        """
        计算趋势（使用移动平均）
        
        Args:
            values: 数值列表
            window: 窗口大小
            
        Returns:
            float: 趋势值（正数表示上升，负数表示下降）
        """
        if len(values) < window * 2:
            return 0
        
        # 计算前半部分和后半部分的平均值
        mid = len(values) // 2
        first_half = np.mean(values[:mid])
        second_half = np.mean(values[mid:])
        
        # 趋势 = (后半部分 - 前半部分) / 前半部分
        if first_half > 0:
            trend = (second_half - first_half) / first_half
        else:
            trend = 0
        
        return trend
    
    def _analyze_smoke_fire_pattern(self, history, pred_dist):
        """
        分析烟雾/火焰的扩散模式
        
        Args:
            history: 检测历史
            pred_dist: 预测分布
            
        Returns:
            dict: 模式分析结果
        """
        pattern = {
            'has_progression': False,  # 是否有从烟雾到火焰的发展
            'smoke_expansion': 0.0,  # 烟雾扩散速率
            'fire_expansion': 0.0,  # 火焰扩散速率
            'pattern_type': 'unknown'  # 模式类型
        }
        
        if not pred_dist:
            return pattern
        
        # 分析预测序列的时序变化
        predictions = [h['lstm_prediction'] for h in history if h['lstm_prediction'] is not None]
        
        if len(predictions) < 30:
            return pattern
        
        # 检查是否有从烟雾(1)到火焰(2)的发展
        smoke_to_fire_transitions = 0
        for i in range(1, len(predictions)):
            if predictions[i-1] == 1 and predictions[i] == 2:
                smoke_to_fire_transitions += 1
        
        if smoke_to_fire_transitions > 3:
            pattern['has_progression'] = True
            pattern['pattern_type'] = 'smoke_to_fire'
        
        # 分析烟雾和火焰各自的扩散
        smoke_areas = []
        fire_areas = []
        
        for h in history:
            if h['features'] is not None and h['lstm_prediction'] is not None:
                area = h['features'][4]
                if h['lstm_prediction'] == 1:  # 烟雾
                    smoke_areas.append(area)
                elif h['lstm_prediction'] == 2:  # 火焰
                    fire_areas.append(area)
        
        # 计算扩散速率（面积增长趋势）
        if len(smoke_areas) > 10:
            smoke_trend = self._calculate_trend(smoke_areas, window=5)
            pattern['smoke_expansion'] = smoke_trend
            if smoke_trend > 0.05:
                pattern['pattern_type'] = 'smoke_spreading'
        
        if len(fire_areas) > 10:
            fire_trend = self._calculate_trend(fire_areas, window=5)
            pattern['fire_expansion'] = fire_trend
            if fire_trend > 0.05:
                pattern['pattern_type'] = 'fire_spreading'
        
        # 如果同时有烟雾和火焰扩散
        if pattern['smoke_expansion'] > 0.05 and pattern['fire_expansion'] > 0.05:
            pattern['pattern_type'] = 'full_fire_development'
        
        return pattern
    
    def _calculate_fire_score(self, detection_rate, pred_dist, conf_mean, 
                              expansion_rate, volatility, movement_rate, pattern):
        """
        计算火灾评分
        
        Args:
            detection_rate: 检测率
            pred_dist: 预测分布
            conf_mean: 平均置信度
            expansion_rate: 扩散速率
            volatility: 波动性
            movement_rate: 运动速率
            pattern: 扩散模式
            
        Returns:
            float: 0-100的评分
        """
        score = 0
        
        # 1. 检测持续性（25分）
        # 真实火灾应该持续被检测到
        if detection_rate > 0.7:
            score += 25
        elif detection_rate > 0.5:
            score += 18
        elif detection_rate > 0.3:
            score += 10
        
        # 2. LSTM预测（35分）
        if pred_dist:
            fire_ratio = pred_dist.get('fire', 0)
            smoke_ratio = pred_dist.get('smoke', 0)
            
            if fire_ratio > 0.5:
                score += 35
            elif fire_ratio > 0.3:
                score += 25
            elif smoke_ratio > 0.5:
                score += 22  # 烟雾也是火灾的重要指标
            elif smoke_ratio > 0.3:
                score += 12
        
        # 3. 置信度（10分）
        if conf_mean > 0.8:
            score += 10
        elif conf_mean > 0.6:
            score += 7
        elif conf_mean > 0.4:
            score += 4
        
        # 4. 扩散趋势（15分）
        # 真实火灾通常会扩散（面积增大）
        if expansion_rate > 0.1:  # 扩散超过10%
            score += 15
        elif expansion_rate > 0.05:
            score += 10
        elif expansion_rate > 0:
            score += 5
        
        # 5. 扩散模式加分（15分）
        if pattern['pattern_type'] == 'full_fire_development':
            score += 15  # 完整的火灾发展过程
        elif pattern['pattern_type'] == 'smoke_to_fire':
            score += 12  # 从烟雾发展到火焰
        elif pattern['pattern_type'] == 'fire_spreading':
            score += 10  # 火焰扩散
        elif pattern['pattern_type'] == 'smoke_spreading':
            score += 8  # 烟雾扩散
        
        # 6. 运动模式（5分）
        # 烟雾向上飘散，火焰跳动，都会有一定运动
        if 0.01 < movement_rate < 0.1:
            score += 5  # 适度运动
        elif movement_rate > 0.1:
            score += 2  # 运动过快可能是误报
        
        # 7. 波动性惩罚（-10分）
        # 打火机等小火焰波动性很高
        if volatility > 0.5:
            score -= 10
        elif volatility > 0.3:
            score -= 5
        
        # 确保分数在0-100之间
        score = max(0, min(100, score))
        
        return score
    
    def _generate_conclusion(self, fire_score, pred_dist, pattern):
        """
        生成结论
        
        Args:
            fire_score: 火灾评分
            pred_dist: 预测分布
            pattern: 扩散模式
            
        Returns:
            dict: 结论信息
        """
        if fire_score >= 70:
            level = 'high'
            message = '⚠️ 高度疑似火灾！建议立即采取行动'
            color = 'red'
        elif fire_score >= 50:
            level = 'medium'
            message = '⚠️ 中度疑似火灾，建议进一步确认'
            color = 'orange'
        elif fire_score >= 30:
            level = 'low'
            message = '⚠️ 低度疑似火灾，可能是误报或小火源'
            color = 'yellow'
        else:
            level = 'safe'
            message = '✓ 未检测到明显火灾迹象'
            color = 'green'
        
        # 添加详细说明
        details = []
        if pred_dist:
            fire_ratio = pred_dist.get('fire', 0)
            smoke_ratio = pred_dist.get('smoke', 0)
            
            if fire_ratio > 0.5:
                details.append(f'检测到大量火焰（{fire_ratio*100:.1f}%）')
            elif fire_ratio > 0.3:
                details.append(f'检测到火焰（{fire_ratio*100:.1f}%）')
            
            if smoke_ratio > 0.5:
                details.append(f'检测到大量烟雾（{smoke_ratio*100:.1f}%）')
            elif smoke_ratio > 0.3:
                details.append(f'检测到烟雾（{smoke_ratio*100:.1f}%）')
        
        # 添加扩散模式说明
        if pattern['pattern_type'] == 'full_fire_development':
            details.append('观察到完整的火灾发展过程（烟雾+火焰扩散）')
        elif pattern['pattern_type'] == 'smoke_to_fire':
            details.append('观察到从烟雾发展到火焰的过程')
        elif pattern['pattern_type'] == 'fire_spreading':
            details.append(f'火焰持续扩散（扩散率: {pattern["fire_expansion"]:.2%}）')
        elif pattern['pattern_type'] == 'smoke_spreading':
            details.append(f'烟雾持续扩散（扩散率: {pattern["smoke_expansion"]:.2%}）')
        
        return {
            'level': level,
            'message': message,
            'color': color,
            'details': details,
            'score': fire_score
        }
    
    def reset(self):
        """重置历史记录"""
        self.detection_history.clear()
    
    def get_summary(self):
        """
        获取简要摘要
        
        Returns:
            str: 摘要文本
        """
        analysis = self.analyze_sequence(min_frames=10)
        
        if analysis['status'] != 'success':
            return analysis['message']
        
        conclusion = analysis['conclusion']
        summary = f"{conclusion['message']}\n"
        summary += f"火灾评分: {conclusion['score']:.1f}/100\n"
        
        if conclusion['details']:
            summary += "详情: " + ", ".join(conclusion['details'])
        
        return summary


if __name__ == "__main__":
    # 测试代码
    print("测试后处理分析器...")
    
    processor = PostProcessor()
    
    # 模拟一些检测结果
    for i in range(50):
        # 模拟火灾场景：持续检测到火焰，面积逐渐增大
        result = {
            'has_detection': True,
            'yolo_detections': [{'class': 'fire'}],
            'lstm_prediction': 2,  # 火焰
            'lstm_confidence': 0.8 + np.random.rand() * 0.15,
            'features': np.array([0.5, 0.5, 0.1, 0.1, 0.01 * (1 + i * 0.02), 1.0, 0.8, 0])
        }
        processor.add_detection(result)
    
    # 分析
    analysis = processor.analyze_sequence()
    
    print("\n分析结果:")
    print(f"总帧数: {analysis['total_frames']}")
    print(f"检测率: {analysis['detection_rate']*100:.1f}%")
    print(f"预测分布: {analysis['prediction_distribution']}")
    print(f"平均置信度: {analysis['confidence']['mean']:.3f}")
    print(f"置信度趋势: {analysis['confidence']['trend']:.3f}")
    print(f"面积扩散率: {analysis['area_analysis']['expansion_rate']:.3f}")
    print(f"波动性: {analysis['area_analysis']['volatility']:.3f}")
    print(f"火灾评分: {analysis['fire_score']:.1f}/100")
    print(f"\n结论: {analysis['conclusion']['message']}")
    
    print("\n✅ 后处理分析器测试完成！")
