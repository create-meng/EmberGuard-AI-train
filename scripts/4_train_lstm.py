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


def run_quick_test(lstm_model, device='cpu'):
    """
    快速测试函数 - 在训练过程中调用
    
    Args:
        lstm_model: LSTM模型
        device: 计算设备
        
    Returns:
        dict: 测试结果
    """
    from pathlib import Path
    import cv2
    import numpy as np
    from emberguard.feature_extractor import FeatureExtractor
    from ultralytics import YOLO
    
    # 加载YOLO模型
    yolo_model = YOLO('runs/detect/train2/weights/best.pt')
    feature_extractor = FeatureExtractor()
    
    # 测试图片
    test_images = [
        'test_picture/1.png',  # 烟雾
        'test_picture/2.jpg',  # 烟雾
        'test_picture/3.jpg',  # 火焰+烟雾
        'test_picture/4.jpg'   # 火焰
    ]
    
    # 已知标签（手动标注）
    # 标注策略：有火焰就标记为火焰(2)，只有烟雾标记为烟雾(1)
    # 这与训练数据的标注策略一致（fire目录→2, smoke目录→1）
    true_labels = [1, 1, 2, 2]  # 0=无火, 1=烟雾, 2=火焰
    
    # 注意：图片3既有火又有烟，但在单标签系统中只能标记为火焰(2)
    # 这是当前模型设计的局限性，未来可以改为多标签分类
    
    correct = 0
    total = 0
    confidences = []
    fire_correct = 0
    fire_total = 0
    
    lstm_model.eval()
    
    for img_path, true_label in zip(test_images, true_labels):
        if not Path(img_path).exists():
            continue
        
        # 读取图片
        img = cv2.imread(img_path)
        
        # YOLO检测
        results = yolo_model(img, verbose=False)
        
        # 提取特征
        features = feature_extractor.get_best_detection(results, img.shape)
        
        # 创建30帧序列
        sequence = np.array([features] * 30)
        
        # LSTM预测
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            outputs = lstm_model(sequence_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # 统计
        total += 1
        if pred_class == true_label:
            correct += 1
        
        confidences.append(confidence)
        
        # 统计Fire类别
        if true_label == 2:
            fire_total += 1
            if pred_class == 2:
                fire_correct += 1
    
    lstm_model.train()
    
    return {
        'image_accuracy': 100 * correct / total if total > 0 else 0,
        'fire_correct': fire_correct,
        'fire_total': fire_total,
        'avg_confidence': np.mean(confidences) if confidences else 0,
        'total_tested': total
    }


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
    device=None,
    resume=False,
    use_focal_loss=True,
    focal_gamma=2.0,
    test_interval=5
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
        resume: 是否从断点继续训练
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
    
    # 计算类别权重（处理类别不平衡）
    label_counts = np.array(metadata['label_distribution'])
    total_samples = label_counts.sum()
    class_weights = total_samples / (len(label_counts) * label_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\n类别权重（处理不平衡）:")
    for i, (name, count, weight) in enumerate(zip(metadata['class_names'], label_counts, class_weights)):
        print(f"  {name} (标签{i}): {count}个样本 ({count/total_samples*100:.1f}%), 权重={weight:.3f}")
    
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
    
    # 创建训练器（使用Focal Loss）
    trainer = LSTMTrainer(model, device=device)
    
    # 选择损失函数
    if use_focal_loss:
        # Focal Loss会自动处理类别不平衡，但仍可以结合类别权重
        # 这里使用较小的权重，让Focal Loss发挥主要作用
        adjusted_weights = torch.FloatTensor([1.0, 1.0, 2.0]).to(device)  # Fire权重降低到2x
        trainer.compile(
            learning_rate=learning_rate, 
            class_weights=adjusted_weights,
            use_focal_loss=True,
            focal_gamma=focal_gamma
        )
        print(f"✅ 使用Focal Loss (gamma={focal_gamma}) + 调整后的类别权重")
        print(f"   Normal: 1.0x, Smoke: 1.0x, Fire: 2.0x")
    else:
        # 传统加权交叉熵
        trainer.compile(learning_rate=learning_rate, class_weights=class_weights)
        print(f"✅ 使用加权交叉熵")
    
    # 验证trainer已正确初始化
    if trainer.criterion is None or trainer.optimizer is None:
        raise RuntimeError("训练器初始化失败：criterion或optimizer为None")
    
    # 添加学习率调度器（每10个epoch降低学习率）
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=10, gamma=0.5)
    print(f"学习率调度: 每10个epoch降低50%")
    
    # 检查是否从断点继续
    start_epoch = 0
    best_val_acc = 0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    checkpoint_path = output_dir / 'checkpoint.pt'
    if resume and checkpoint_path.exists():
        print(f"\n从断点继续训练: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 恢复模型和优化器状态
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        best_epoch = checkpoint['best_epoch']
        history = checkpoint['history']
        
        # 检查损失函数一致性
        if 'use_focal_loss' in checkpoint:
            saved_focal = checkpoint['use_focal_loss']
            if saved_focal != use_focal_loss:
                print(f"⚠️  警告: 断点使用的损失函数与当前不同")
                print(f"   断点: {'Focal Loss' if saved_focal else 'CrossEntropyLoss'}")
                print(f"   当前: {'Focal Loss' if use_focal_loss else 'CrossEntropyLoss'}")
                print(f"   将使用当前设置继续训练")
        
        print(f"从 Epoch {start_epoch} 继续训练")
        print(f"当前最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    else:
        if resume:
            print("\n⚠️  未找到断点文件，从头开始训练")
    
    # 训练
    print(f"\n开始训练 (epochs={start_epoch+1}-{epochs})...")
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print("=" * 60)
    
    # 创建日志文件
    log_file = output_dir / 'training.log'
    log_mode = 'a' if (resume and log_file.exists()) else 'w'
    
    with open(log_file, log_mode, encoding='utf-8') as f:
        if log_mode == 'w':
            f.write(f"EmberGuard AI - LSTM训练日志\n")
            f.write(f"{'='*60}\n")
            f.write(f"开始时间: {__import__('datetime').datetime.now()}\n")
            f.write(f"设备: {device}\n")
            f.write(f"训练集: {len(train_dataset)} 样本\n")
            f.write(f"验证集: {len(val_dataset)} 样本\n")
            f.write(f"批次大小: {batch_size}\n")
            f.write(f"学习率: {learning_rate}\n")
            f.write(f"训练轮数: {epochs}\n")
            f.write(f"{'='*60}\n\n")
        else:
            f.write(f"\n{'='*60}\n")
            f.write(f"从断点继续训练: {__import__('datetime').datetime.now()}\n")
            f.write(f"从 Epoch {start_epoch} 继续\n")
            f.write(f"{'='*60}\n\n")
    
    import time
    start_time = time.time()
    
    try:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            
            # 训练（显示进度条）
            train_loss, train_acc = trainer.train_epoch(train_loader, show_progress=True)
            
            # 验证（显示进度条）
            val_loss, val_acc = trainer.validate(val_loader, show_progress=True)
            
            # 记录
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # 计算时间
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (epoch - start_epoch + 1)) * (epochs - epoch - 1)
            
            # 更新学习率
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            # 打印到控制台
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s ETA: {eta/60:.1f}min")
            
            # 写入日志文件
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Epoch {epoch+1}/{epochs}\n")
                f.write(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%\n")
                f.write(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%\n")
                f.write(f"  LR: {current_lr:.6f}\n")
                f.write(f"  Time: {epoch_time:.1f}s, Elapsed: {elapsed_time/60:.1f}min, ETA: {eta/60:.1f}min\n")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                trainer.save_model(output_dir / 'best.pt')
                msg = f"  ✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)"
                print(msg)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"  {msg}\n")
            
            # 保存断点（每个epoch都保存，用于断点续训）
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'history': history,
                'use_focal_loss': use_focal_loss,
                'focal_gamma': focal_gamma if use_focal_loss else None
            }
            torch.save(checkpoint, output_dir / 'checkpoint.pt')
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("\n")
            
            # 定期测试
            if test_interval > 0 and (epoch + 1) % test_interval == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1} - 运行测试")
                print(f"{'='*60}")
                
                try:
                    # 运行快速测试
                    test_results = run_quick_test(lstm_model=trainer.model, device=device)
                    
                    # 记录测试结果到日志
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"  测试结果 (Epoch {epoch+1}):\n")
                        f.write(f"    图片测试准确率: {test_results['image_accuracy']:.1f}%\n")
                        f.write(f"    Fire识别: {test_results['fire_correct']}/{test_results['fire_total']}\n")
                        f.write(f"    平均置信度: {test_results['avg_confidence']:.3f}\n")
                        f.write("\n")
                    
                    print(f"✅ 测试完成 - 准确率: {test_results['image_accuracy']:.1f}%")
                    
                except Exception as e:
                    print(f"⚠️  测试失败: {e}")
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"  测试失败: {e}\n\n")
                
                print(f"{'='*60}\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print(f"已保存断点到: {output_dir / 'checkpoint.pt'}")
        print(f"使用 --resume 参数可以继续训练")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"训练被用户中断: {__import__('datetime').datetime.now()}\n")
            f.write(f"已完成 {epoch+1} 个epoch\n")
            f.write(f"{'='*60}\n")
        return
    
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        print(f"已保存断点到: {output_dir / 'checkpoint.pt'}")
        print(f"使用 --resume 参数可以继续训练")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"训练出错: {__import__('datetime').datetime.now()}\n")
            f.write(f"错误信息: {e}\n")
            f.write(f"{'='*60}\n")
        raise
    
    total_time = time.time() - start_time
    
    print("=" * 60)
    print(f"\n训练完成!")
    print(f"总耗时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # 写入最终结果到日志
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"训练完成\n")
        f.write(f"结束时间: {__import__('datetime').datetime.now()}\n")
        f.write(f"总耗时: {total_time/60:.1f} 分钟\n")
        f.write(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})\n")
        f.write(f"{'='*60}\n")
    
    print(f"\n训练日志已保存: {log_file}")
    
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
    parser.add_argument('--resume', action='store_true',
                       help='从断点继续训练')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                       help='使用Focal Loss（默认开启）')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss的gamma参数（默认2.0）')
    parser.add_argument('--test_interval', type=int, default=5,
                       help='测试间隔（每N个epoch测试一次，0表示不测试）')
    
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
    
    # 自动检测断点
    checkpoint_path = Path(args.output_dir) / 'checkpoint.pt'
    if checkpoint_path.exists() and not args.resume:
        print(f"\n⚠️  检测到断点文件: {checkpoint_path}")
        print("是否从断点继续训练？")
        print("  1. 是 - 继续训练")
        print("  2. 否 - 从头开始（会覆盖之前的训练）")
        choice = input("请选择 (1/2): ").strip()
        
        if choice == '1':
            args.resume = True
            print("✅ 将从断点继续训练")
        else:
            print("⚠️  将从头开始训练")
            # 备份旧的断点
            import shutil
            backup_path = Path(args.output_dir) / f'checkpoint_backup_{__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
            shutil.copy(checkpoint_path, backup_path)
            print(f"已备份旧断点到: {backup_path}")
    
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
        device=args.device,
        resume=args.resume,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        test_interval=args.test_interval
    )


if __name__ == "__main__":
    main()
