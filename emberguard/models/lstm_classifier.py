"""
LSTM时序分类器 - 时间特征分析
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple
import os


class LSTMFireClassifier:
    """
    基于LSTM的火灾时序分类器
    分析连续帧的特征序列，判断是否为真实火灾
    """
    
    def __init__(self, model_path: str = None, seq_length: int = 30, num_features: int = 11):
        """
        初始化LSTM分类器
        
        Args:
            model_path: 预训练模型路径（如果为None则创建新模型）
            seq_length: 序列长度（帧数）
            num_features: 每帧的特征数量
        """
        self.seq_length = seq_length
        self.num_features = num_features
        self.class_names = ['no_fire', 'smoke', 'fire']
        
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"✅ 已加载LSTM模型: {model_path}")
        else:
            self.model = self._build_model()
            print("⚠️  创建了新的LSTM模型（未训练）")
    
    def _build_model(self) -> keras.Model:
        """
        构建LSTM模型架构
        
        Returns:
            Keras模型
        """
        model = keras.Sequential([
            # 输入层
            layers.Input(shape=(self.seq_length, self.num_features)),
            
            # LSTM层1 - 提取时序特征
            layers.LSTM(128, return_sequences=True, dropout=0.3),
            layers.BatchNormalization(),
            
            # LSTM层2 - 深层时序特征
            layers.LSTM(64, return_sequences=False, dropout=0.3),
            layers.BatchNormalization(),
            
            # 全连接层
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            
            # 输出层 - 3分类（no_fire, smoke, fire）
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, feature_sequence: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        预测特征序列的类别
        
        Args:
            feature_sequence: 特征序列 shape=(seq_length, num_features)
            
        Returns:
            (预测类别名称, 置信度, 所有类别概率)
        """
        # 确保输入形状正确
        if feature_sequence.shape[0] < self.seq_length:
            # 如果序列不够长，用零填充
            padding = np.zeros((self.seq_length - feature_sequence.shape[0], self.num_features))
            feature_sequence = np.vstack([padding, feature_sequence])
        elif feature_sequence.shape[0] > self.seq_length:
            # 如果序列太长，只取最后seq_length帧
            feature_sequence = feature_sequence[-self.seq_length:]
        
        # 添加batch维度
        input_seq = np.expand_dims(feature_sequence, axis=0)
        
        # 预测
        predictions = self.model.predict(input_seq, verbose=0)[0]
        
        # 获取最高概率的类别
        class_idx = int(np.argmax(predictions))
        class_name = self.class_names[class_idx]
        confidence = float(predictions[class_idx])
        
        return class_name, confidence, predictions
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> keras.callbacks.History:
        """
        训练LSTM模型
        
        Args:
            X_train: 训练特征 shape=(n_samples, seq_length, num_features)
            y_train: 训练标签 shape=(n_samples, num_classes) one-hot编码
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            训练历史
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        self.model.save(save_path)
        print(f"✅ 模型已保存到: {save_path}")
    
    def get_model_summary(self):
        """打印模型结构"""
        return self.model.summary()
