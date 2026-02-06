"""
准备LSTM训练数据
从视频中提取特征序列并标注
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import json
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from emberguard.feature_extractor import FeatureExtractor
from ultralytics import YOLO


class LSTMDataPreparer:
    """LSTM数据准备器"""
    
    def __init__(self, yolo_model_path, sequence_length=30):
        """
        初始化
        
        Args:
            yolo_model_path: YOLO模型路径
            sequence_length: 序列长度（帧数）
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.feature_extractor = FeatureExtractor()
        self.sequence_length = sequence_length
        
    def extract_features_from_video(self, video_path, stride=1):
        """
        从视频提取特征序列
        
        Args:
            video_path: 视频路径
            stride: 采样步长（每隔stride帧提取一次）
            
        Returns:
            list: 特征序列列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"处理视频: {video_path}")
        print(f"总帧数: {total_frames}")
        
        features_list = []
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="提取特征") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按步长采样
                if frame_idx % stride == 0:
                    # YOLO检测
                    results = self.yolo_model(frame, verbose=False)
                    
                    # 提取特征
                    features = self.feature_extractor.get_best_detection(results, frame.shape)
                    features_list.append(features)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        print(f"提取了 {len(features_list)} 个特征向量")
        return features_list
    
    def create_sequences(self, features_list, label):
        """
        创建训练序列
        
        Args:
            features_list: 特征列表
            label: 标签 (0=无火, 1=烟雾, 2=火焰)
            
        Returns:
            tuple: (sequences, labels)
        """
        sequences = []
        labels = []
        
        # 滑动窗口创建序列
        for i in range(len(features_list) - self.sequence_length + 1):
            seq = features_list[i:i + self.sequence_length]
            sequences.append(seq)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def prepare_dataset(self, video_list, output_dir):
        """
        准备完整数据集
        
        Args:
            video_list: 视频列表 [(video_path, label), ...]
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_sequences = []
        all_labels = []
        
        for video_path, label in video_list:
            print(f"\n处理: {video_path} (标签: {label})")
            
            # 提取特征
            features = self.extract_features_from_video(video_path)
            
            # 创建序列
            sequences, labels = self.create_sequences(features, label)
            
            all_sequences.append(sequences)
            all_labels.append(labels)
            
            print(f"生成 {len(sequences)} 个序列")
        
        # 合并所有数据
        all_sequences = np.concatenate(all_sequences, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"\n总序列数: {len(all_sequences)}")
        print(f"序列形状: {all_sequences.shape}")
        print(f"标签分布: {np.bincount(all_labels)}")
        
        # 保存数据
        np.save(output_dir / 'sequences.npy', all_sequences)
        np.save(output_dir / 'labels.npy', all_labels)
        
        # 保存元数据
        metadata = {
            'num_sequences': len(all_sequences),
            'sequence_length': self.sequence_length,
            'feature_dim': 8,
            'num_classes': 3,
            'class_names': ['无火', '烟雾', '火焰'],
            'label_distribution': np.bincount(all_labels).tolist()
        }
        
        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n数据已保存到: {output_dir}")
        print(f"- sequences.npy: {all_sequences.shape}")
        print(f"- labels.npy: {all_labels.shape}")
        print(f"- metadata.json")


def main():
    """主函数"""
    print("=" * 60)
    print("LSTM训练数据准备工具")
    print("=" * 60)
    
    # 初始化
    preparer = LSTMDataPreparer(
        yolo_model_path='runs/detect/train2/weights/best.pt',
        sequence_length=30
    )
    
    # 示例：准备数据集
    # 注意：需要准备标注好的视频数据
    print("\n⚠️  数据准备说明:")
    print("1. 准备视频数据，分为三类：")
    print("   - 无火视频（标签0）")
    print("   - 烟雾视频（标签1）")
    print("   - 火焰视频（标签2）")
    print("\n2. 创建视频列表，格式：")
    print("   video_list = [")
    print("       ('path/to/no_fire.mp4', 0),")
    print("       ('path/to/smoke.mp4', 1),")
    print("       ('path/to/fire.mp4', 2),")
    print("   ]")
    print("\n3. 调用 preparer.prepare_dataset(video_list, 'datasets/lstm_data')")
    
    # 示例代码（需要实际视频数据）
    """
    video_list = [
        ('videos/no_fire_1.mp4', 0),
        ('videos/no_fire_2.mp4', 0),
        ('videos/smoke_1.mp4', 1),
        ('videos/smoke_2.mp4', 1),
        ('videos/fire_1.mp4', 2),
        ('videos/fire_2.mp4', 2),
    ]
    
    preparer.prepare_dataset(video_list, 'datasets/lstm_data')
    """
    
    print("\n✅ 数据准备工具已就绪")
    print("请根据实际视频数据修改代码并运行")


if __name__ == "__main__":
    main()
