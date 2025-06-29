#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频特征提取和工具函数模块

主要功能：
1. 音频文件加载和预处理
2. MFCC特征提取
3. 数据标准化和处理
"""

import librosa
import numpy as np
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """
    音频特征提取器类
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 duration: float = 3.0):
        """
        初始化特征提取器
        
        Args:
            sample_rate: 采样率
            n_mfcc: MFCC特征维数
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            duration: 音频片段时长（秒）
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.max_length = int(sample_rate * duration)
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频数据数组
        """
        try:
            # 加载音频文件
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # 确保音频长度一致
            if len(audio) < self.max_length:
                # 如果音频太短，用零填充
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
            else:
                # 如果音频太长，截取前面部分
                audio = audio[:self.max_length]
                
            return audio
            
        except Exception as e:
            print(f"加载音频文件 {file_path} 时出错: {e}")
            return np.zeros(self.max_length)
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            audio: 音频数据
            
        Returns:
            MFCC特征向量（统计特征）
        """
        try:
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # 计算统计特征：均值和标准差
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # 合并特征向量
            features = np.concatenate([mfcc_mean, mfcc_std])
            
            return features
            
        except Exception as e:
            print(f"提取MFCC特征时出错: {e}")
            return np.zeros(self.n_mfcc * 2)
    
    def extract_additional_features(self, audio: np.ndarray) -> np.ndarray:
        """
        提取额外的音频特征
        
        Args:
            audio: 音频数据
            
        Returns:
            额外特征向量
        """
        try:
            # 零交叉率
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # 频谱质心
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            
            # 频谱带宽
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate))
            
            # 频谱滚降
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
            
            # RMS能量
            rms = np.mean(librosa.feature.rms(y=audio))
            
            return np.array([zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff, rms])
            
        except Exception as e:
            print(f"提取额外特征时出错: {e}")
            return np.zeros(5)
    
    def extract_features_from_file(self, file_path: str, include_additional: bool = False) -> np.ndarray:
        """
        从音频文件提取完整特征
        
        Args:
            file_path: 音频文件路径
            include_additional: 是否包含额外特征
            
        Returns:
            特征向量
        """
        # 加载音频
        audio = self.load_audio(file_path)
        
        # 提取MFCC特征
        mfcc_features = self.extract_mfcc_features(audio)
        
        if include_additional:
            # 提取额外特征
            additional_features = self.extract_additional_features(audio)
            # 合并所有特征
            features = np.concatenate([mfcc_features, additional_features])
        else:
            features = mfcc_features
            
        return features
    
    def extract_features_from_audio(self, audio: np.ndarray, include_additional: bool = False) -> np.ndarray:
        """
        从音频数据提取完整特征
        
        Args:
            audio: 音频数据
            include_additional: 是否包含额外特征
            
        Returns:
            特征向量
        """
        # 提取MFCC特征
        mfcc_features = self.extract_mfcc_features(audio)
        
        if include_additional:
            # 提取额外特征
            additional_features = self.extract_additional_features(audio)
            # 合并所有特征
            features = np.concatenate([mfcc_features, additional_features])
        else:
            features = mfcc_features
            
        return features


def get_frequency_category(frequency: float) -> str:
    """
    根据频率值确定类别
    
    Args:
        frequency: 频率值（Hz）
        
    Returns:
        频率类别字符串
    """
    if frequency < 1000:
        return 'low_freq'
    elif frequency < 4000:
        return 'mid_freq'
    else:
        return 'high_freq'


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    特征标准化
    
    Args:
        features: 特征矩阵
        
    Returns:
        标准化后的特征矩阵
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    return normalized_features, scaler


def print_feature_info(features: np.ndarray, labels: List[str]):
    """
    打印特征信息
    
    Args:
        features: 特征矩阵
        labels: 标签列表
    """
    print(f"特征矩阵形状: {features.shape}")
    print(f"样本数量: {len(labels)}")
    print(f"特征维度: {features.shape[1]}")
    
    # 统计各类别样本数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n各类别样本数量:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")