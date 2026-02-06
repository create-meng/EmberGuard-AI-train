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


def load_video_list_from_organized():
    """从整理好的目录加载视频列表"""
    from pathlib import Path
    
    base_dir = Path("datasets/fire_videos_organized")
    video_list = []
    
    # 加载火灾视频（标签2）
    fire_dir = base_dir / "fire"
    if fire_dir.exists():
        for video_file in fire_dir.glob("*"):
            if video_file.suffix.lower() in ['.avi', '.mp4', '.mov']:
                video_list.append((str(video_file), 2))
    
    # 加载烟雾视频（标签1）
    smoke_dir = base_dir / "smoke"
    if smoke_dir.exists():
        for video_file in smoke_dir.glob("*"):
            if video_file.suffix.lower() in ['.avi', '.mp4', '.mov']:
                video_list.append((str(video_file), 1))
    
    # 加载正常视频（标签0）
    normal_dir = base_dir / "normal"
    if normal_dir.exists():
        for video_file in normal_dir.glob("*"):
            if video_file.suffix.lower() in ['.avi', '.mp4', '.mov']:
                video_list.append((str(video_file), 0))
    
    return video_list


def main():
    """主函数"""
    print("=" * 60)
    print("LSTM训练数据准备工具")
    print("=" * 60)
    
    # 检查整理好的数据是否存在
    from pathlib import Path
    organized_dir = Path("datasets/fire_videos_organized")
    
    if not organized_dir.exists():
        print("\n❌ 错误: 未找到整理好的数据目录")
        print("请先运行: python scripts/organize_downloaded_data.py")
        return
    
    # 加载视频列表
    print("\n📂 加载整理好的视频数据...")
    video_list = load_video_list_from_organized()
    
    if not video_list:
        print("❌ 错误: 未找到视频文件")
        return
    
    # 统计
    fire_count = sum(1 for _, label in video_list if label == 2)
    smoke_count = sum(1 for _, label in video_list if label == 1)
    normal_count = sum(1 for _, label in video_list if label == 0)
    
    print(f"\n找到视频:")
    print(f"  火灾视频: {fire_count}")
    print(f"  烟雾视频: {smoke_count}")
    print(f"  正常视频: {normal_count}")
    print(f"  总计: {len(video_list)}")
    
    # 初始化
    print("\n🔧 初始化特征提取器...")
    preparer = LSTMDataPreparer(
        yolo_model_path='runs/detect/train2/weights/best.pt',
        sequence_length=30
    )
    
    # 准备数据集
    print("\n🚀 开始准备训练数据...")
    print("这可能需要30-60分钟，请耐心等待...")
    print()
    
    try:
        preparer.prepare_dataset(video_list, 'datasets/lstm_data')
        print("\n" + "=" * 60)
        print("✅ 数据准备完成！")
        print("=" * 60)
        print("\n下一步:")
        print("  python scripts/4_train_lstm.py --data_dir datasets/lstm_data --epochs 50")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
