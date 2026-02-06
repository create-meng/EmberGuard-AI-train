"""
EmberGuard AI - LSTM时序分析模块
"""

__version__ = "0.1.0"
__author__ = "EmberGuard Team"

from .feature_extractor import FeatureExtractor
from .lstm_model import LSTMFireClassifier
from .pipeline import FireDetectionPipeline

__all__ = [
    'FeatureExtractor',
    'LSTMFireClassifier',
    'FireDetectionPipeline'
]
