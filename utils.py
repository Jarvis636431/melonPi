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

# 导入新的模块
from config_manager import config_manager
from logger_setup import setup_logger
from performance_monitor import monitor_performance, PerformanceLogger


class AudioFeatureExtractor:
    """
    音频特征提取器类
    """
    
    def __init__(self, 
                 sample_rate: Optional[int] = None,
                 n_mfcc: Optional[int] = None,
                 n_fft: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 duration: Optional[float] = None):
        """
        初始化特征提取器
        
        Args:
            sample_rate: 采样率（可选，从配置文件读取）
            n_mfcc: MFCC特征维数（可选，从配置文件读取）
            n_fft: FFT窗口大小（可选，从配置文件读取）
            hop_length: 跳跃长度（可选，从配置文件读取）
            duration: 音频片段时长（秒）（可选，从配置文件读取）
        """
        # 初始化日志器和性能监控器
        self.logger = setup_logger('audio_feature_extractor')
        self.performance_logger = PerformanceLogger('audio_feature_extraction')
        
        # 从配置文件获取参数，如果没有提供的话
        self.sample_rate = sample_rate or config_manager.get('audio.sample_rate', 22050)
        self.n_mfcc = n_mfcc or config_manager.get('audio.n_mfcc', 13)
        self.n_fft = n_fft or config_manager.get('audio.n_fft', 2048)
        self.hop_length = hop_length or config_manager.get('audio.hop_length', 512)
        self.duration = duration or config_manager.get('audio.duration', 3.0)
        self.max_length = int(self.sample_rate * self.duration)
        
        self.logger.info(f"音频特征提取器初始化完成 - 采样率: {self.sample_rate}, MFCC维数: {self.n_mfcc}, 时长: {self.duration}秒")
        
        # 记录性能信息
        self.performance_logger.log_performance({
            'operation': 'feature_extractor_init',
            'sample_rate': self.sample_rate,
            'n_mfcc': self.n_mfcc,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'duration': self.duration,
            'max_length': self.max_length
        })
    
    @monitor_performance("load_audio", log_args=True)
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频数据数组
        """
        try:
            self.logger.debug(f"开始加载音频文件: {file_path}")
            
            # 加载音频文件
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # 确保音频长度一致
            if len(audio) < self.max_length:
                # 如果音频太短，用零填充
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
                self.logger.debug(f"音频文件 {file_path} 长度不足，已填充至 {self.max_length} 样本")
            else:
                # 如果音频太长，截取前面部分
                audio = audio[:self.max_length]
                self.logger.debug(f"音频文件 {file_path} 长度过长，已截取至 {self.max_length} 样本")
            
            self.logger.info(f"成功加载音频文件: {file_path}, 长度: {len(audio)} 样本")
            return audio
            
        except Exception as e:
            self.logger.error(f"加载音频文件 {file_path} 时出错: {e}")
            return np.zeros(self.max_length)
    
    @monitor_performance("extract_mfcc_features")
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            audio: 音频数据
            
        Returns:
            MFCC特征向量（统计特征）
        """
        try:
            self.logger.debug(f"开始提取MFCC特征，音频长度: {len(audio)}")
            
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
            
            self.logger.debug(f"MFCC特征提取完成，特征维度: {len(features)}")
            return features
            
        except Exception as e:
            self.logger.error(f"提取MFCC特征时出错: {e}")
            return np.zeros(self.n_mfcc * 2)
    
    @monitor_performance("extract_additional_features")
    def extract_additional_features(self, audio: np.ndarray) -> np.ndarray:
        """
        提取额外的音频特征
        
        Args:
            audio: 音频数据
            
        Returns:
            额外特征向量
        """
        try:
            self.logger.debug("开始提取额外音频特征")
            
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
            
            features = np.array([zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff, rms])
            
            self.logger.debug(f"额外特征提取完成: ZCR={zcr:.4f}, 质心={spectral_centroid:.2f}, 带宽={spectral_bandwidth:.2f}, 滚降={spectral_rolloff:.2f}, RMS={rms:.4f}")
            return features
            
        except Exception as e:
            self.logger.error(f"提取额外特征时出错: {e}")
            return np.zeros(5)
    
    @monitor_performance("extract_features_from_file", log_args=True)
    def extract_features_from_file(self, file_path: str, include_additional: bool = False) -> np.ndarray:
        """
        从音频文件提取完整特征
        
        Args:
            file_path: 音频文件路径
            include_additional: 是否包含额外特征
            
        Returns:
            特征向量
        """
        self.logger.debug(f"开始从文件提取特征: {file_path}, 包含额外特征: {include_additional}")
        
        # 加载音频
        audio = self.load_audio(file_path)
        
        # 提取MFCC特征
        mfcc_features = self.extract_mfcc_features(audio)
        
        if include_additional:
            # 提取额外特征
            additional_features = self.extract_additional_features(audio)
            # 合并所有特征
            features = np.concatenate([mfcc_features, additional_features])
            self.logger.debug(f"特征提取完成，总维度: {len(features)} (MFCC: {len(mfcc_features)}, 额外: {len(additional_features)})")
        else:
            features = mfcc_features
            self.logger.debug(f"MFCC特征提取完成，维度: {len(features)}")
            
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
    # 从配置文件获取频率阈值
    low_threshold = config_manager.get('frequency_categories.low_freq.max_freq', 1000)
    mid_threshold = config_manager.get('frequency_categories.mid_freq.max_freq', 4000)
    
    if frequency < low_threshold:
        return 'low_freq'
    elif frequency < mid_threshold:
        return 'mid_freq'
    else:
        return 'high_freq'


@monitor_performance
def normalize_features(features: np.ndarray, scaler=None) -> tuple:
    """
    特征标准化
    
    Args:
        features: 特征数组
        scaler: 已训练的标准化器（可选）
    
    Returns:
        标准化后的特征和标准化器
    """
    from sklearn.preprocessing import StandardScaler
    
    logger = setup_logger("normalize_features")
    logger.debug(f"开始特征标准化，输入特征形状: {features.shape}")
    
    if scaler is None:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        logger.debug("使用新的StandardScaler进行fit_transform")
    else:
        normalized_features = scaler.transform(features)
        logger.debug("使用已有的StandardScaler进行transform")
    
    logger.info(f"特征标准化完成，输出特征形状: {normalized_features.shape}")
    return normalized_features, scaler


@monitor_performance
def create_frequency_labels(frequencies: list, labels: list = None) -> dict:
    """
    创建频率标签映射
    
    Args:
        frequencies: 频率列表
        labels: 标签列表（可选）
    
    Returns:
        频率到标签的映射字典
    """
    logger = setup_logger("create_frequency_labels")
    logger.debug(f"创建频率标签映射，频率数量: {len(frequencies)}")
    
    if labels is None:
        labels = [f"freq_{freq}Hz" for freq in frequencies]
        logger.debug("使用默认标签格式")
    else:
        logger.debug(f"使用自定义标签，标签数量: {len(labels)}")
    
    mapping = dict(zip(frequencies, labels))
    logger.info(f"频率标签映射创建完成: {mapping}")
    return mapping


def print_feature_info(features: np.ndarray, labels: List[str]):
    """
    打印特征信息
    
    Args:
        features: 特征矩阵
        labels: 标签列表
    """
    logger = setup_logger("feature_info")
    
    print(f"特征矩阵形状: {features.shape}")
    print(f"样本数量: {len(labels)}")
    print(f"特征维度: {features.shape[1]}")
    
    logger.info(f"特征信息 - 矩阵形状: {features.shape}, 样本数量: {len(labels)}, 特征维度: {features.shape[1]}")
    
    # 统计各类别样本数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n各类别样本数量:")
    category_stats = {}
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")
        category_stats[label] = count
    
    logger.info(f"各类别样本统计: {category_stats}")