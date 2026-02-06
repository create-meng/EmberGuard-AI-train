"""
训练LSTM模型
"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from emberguard.lstm_model import LSTMFireClassifier, LSTMTrainer


class FireSequenceDataset(Dataset):
    """火灾序列数据集"""
    
    def __init__(self, sequences, labels):
        """
        初始化
        
        Args:
            sequences: 特征序列 (N, seq_len, feature_dim)
            labels: 标签 (N,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_dataset(data_dir):
    """
    加载数据集
    
    Args:
        data_dir: 数据目录
        
    Returns:
        tuple: (sequences, labels, metadata)
    """
    data_dir = Path(data_dir)
    
    sequences = np.load(data_dir / 'sequences.npy')
    labels = np.load(data_dir / 'labels.npy')
    
    with open(data_dir / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return sequences, labels, metadata


def train_lstm_model(
    data_dir,
    output_dir='models/lstm',
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    batch_size=32,
    epochs=50,
    learning_rate=0.001,
    val_split=0.2,
    device=None
):
    """
    训练LSTM模型
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        dropout: Dropout比例
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        val_split: 验证集比例
        device: 计算设备
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    sequences, labels, metadata = load_dataset(data_dir)
    
    print(f"数据集信息:")
    print(f"- 序列数: {len(sequences)}")
    print(f"- 序列长度: {metadata['sequence_length']}")
    print(f"- 特征维度: {metadata['feature_dim']}")
    print(f"- 类别数: {metadata['num_classes']}")
    print(f"- 标签分布: {metadata['label_distribution']}")
    
    # 创建数据集
    dataset = FireSequenceDataset(sequences, labels)
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n数据划分:")
    print(f"- 训练集: {len(train_dataset)} 样本")
    print(f"- 验证集: {len(val_dataset)} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print("\n创建模型...")
    model = LSTMFireClassifier(
        input_size=metadata['feature_dim'],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=metadata['num_classes'],
        dropout=dropout
    )
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters())} 个")
    
    # 创建训练器
    trainer = LSTMTrainer(model, device=device)
    trainer.compile(learning_rate=learning_rate)
    
    # 训练
    print(f"\n开始训练 (epochs={epochs})...")
    print("=" * 60)
    
    best_val_acc = 0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # 验证
        val_loss, val_acc = trainer.validate(val_loader)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            trainer.save_model(output_dir / 'best.pt')
            print(f"  ✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
    
    print("=" * 60)
    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # 保存最终模型
    trainer.save_model(output_dir / 'last.pt')
    
    # 保存训练历史
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 保存训练配置
    config = {
        'data_dir': str(data_dir),
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'val_split': val_split,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n模型已保存到: {output_dir}")
    print(f"- best.pt: 最佳模型")
    print(f"- last.pt: 最终模型")
    print(f"- history.json: 训练历史")
    print(f"- config.json: 训练配置")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练LSTM火灾分类模型')
    parser.add_argument('--data_dir', type=str, default='datasets/lstm_data',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='models/lstm',
                       help='输出目录')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout比例')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LSTM火灾分类模型训练")
    print("=" * 60)
    
    # 检查数据是否存在
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\n❌ 错误: 数据目录不存在: {data_dir}")
        print("\n请先运行 prepare_lstm_data.py 准备训练数据")
        return
    
    # 训练模型
    train_lstm_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        val_split=args.val_split,
        device=args.device
    )


if __name__ == "__main__":
    main()
