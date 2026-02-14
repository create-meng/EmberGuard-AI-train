"""
LSTM火灾分类器 - 基于时序特征的火灾识别
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss - 解决类别不平衡问题
    
    论文: Focal Loss for Dense Object Detection (Lin et al., 2017)
    
    公式: FL(pt) = -α(1-pt)^γ * log(pt)
    
    参数:
        alpha: 类别权重，平衡正负样本
        gamma: 聚焦参数，降低易分类样本的权重
               gamma=0 时退化为交叉熵
               gamma越大，对难分类样本的关注越多
    
    优势:
        1. 自动降低易分类样本的loss贡献
        2. 自动提高难分类样本的loss贡献
        3. 不需要手动调整类别权重
        4. 训练更稳定
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 类别权重 (可选)，shape: (num_classes,)
            gamma: 聚焦参数，推荐值2.0
            reduction: 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        计算Focal Loss
        
        Args:
            inputs: 模型输出 (batch, num_classes)
            targets: 真实标签 (batch,)
            
        Returns:
            loss值
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算pt (预测正确的概率)
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # 返回
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LSTMFireClassifier(nn.Module):
    """
    LSTM火灾分类器
    
    输入: (batch, seq_len, 8) - 时序特征序列
    输出: (batch, 3) - [无火, 烟雾, 火焰]
    """
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度 (默认8)
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            num_classes: 分类数量 (默认3: 无火/烟雾/火焰)
            dropout: Dropout比例
        """
        super(LSTMFireClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, seq_len, input_size)
            
        Returns:
            (batch, num_classes)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        out = self.dropout(last_output)
        
        # 全连接层
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def predict(self, x):
        """
        预测（带softmax）
        
        Args:
            x: (batch, seq_len, input_size) 或 (seq_len, input_size)
            
        Returns:
            预测类别和概率
        """
        self.eval()
        with torch.no_grad():
            # 转换为tensor
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            
            # 处理单个样本
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            
            # 前向传播
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            
            return pred_class.cpu().numpy(), probs.cpu().numpy()


class LSTMTrainer:
    """LSTM训练器"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        
        Args:
            model: LSTM模型
            device: 训练设备
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = None
        self.optimizer = None
        
    def compile(self, learning_rate=0.001, weight_decay=1e-5, class_weights=None, use_focal_loss=False, focal_gamma=2.0):
        """
        配置优化器和损失函数
        
        Args:
            learning_rate: 学习率
            weight_decay: 权重衰减
            class_weights: 类别权重（用于处理类别不平衡）
            use_focal_loss: 是否使用Focal Loss（推荐）
            focal_gamma: Focal Loss的gamma参数（默认2.0）
        """
        # 设置损失函数
        if use_focal_loss:
            # 使用Focal Loss（推荐）
            print(f"使用Focal Loss (gamma={focal_gamma})")
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            # 使用传统交叉熵（支持类别权重）
            if class_weights is not None:
                print("使用加权交叉熵")
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                print("使用标准交叉熵")
                self.criterion = nn.CrossEntropyLoss()
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
    def train_epoch(self, train_loader, show_progress=False):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            show_progress: 是否显示进度条
            
        Returns:
            平均损失和准确率
        """
        from tqdm import tqdm
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 创建进度条
        if show_progress:
            pbar = tqdm(train_loader, desc="Training", leave=False)
        else:
            pbar = train_loader
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # 更新进度条
            if show_progress:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, show_progress=False):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            show_progress: 是否显示进度条
            
        Returns:
            平均损失和准确率
        """
        from tqdm import tqdm
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # 创建进度条
        if show_progress:
            pbar = tqdm(val_loader, desc="Validation", leave=False)
        else:
            pbar = val_loader
        
        with torch.no_grad():
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # 更新进度条
                if show_progress:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * correct / total:.2f}%'
                    })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_classes': self.model.num_classes
            }
        }, path)
        print(f"模型已保存到: {path}")
    
    @staticmethod
    def load_model(path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        加载模型
        
        Args:
            path: 模型路径
            device: 设备
            
        Returns:
            加载的模型
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = LSTMFireClassifier(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        if not os.environ.get('SILENT_MODE'):
            print(f"模型已从 {path} 加载")
        return model


if __name__ == "__main__":
    # 测试代码
    print("测试LSTM模型...")
    
    # 创建模型
    model = LSTMFireClassifier(
        input_size=8,
        hidden_size=128,
        num_layers=2,
        num_classes=3
    )
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters())} 个")
    
    # 测试前向传播
    batch_size = 4
    seq_len = 30
    input_size = 8
    
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试预测
    pred_class, probs = model.predict(x)
    print(f"预测类别: {pred_class}")
    print(f"预测概率: {probs}")
    
    print("\n✅ LSTM模型测试完成！")
