#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派实时音频频率分类推理脚本

功能：
1. 实时录制音频
2. 提取音频特征
3. 使用训练好的模型进行分类
4. 输出分类结果

使用方法：
    python inference_pi.py

注意：
    - 确保已安装所需依赖
    - 确保模型文件存在于model目录中
    - 确保音频设备正常工作
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import joblib
from utils import AudioFeatureExtractor
import warnings
warnings.filterwarnings('ignore')


class RealTimeFrequencyClassifier:
    """
    实时频率分类器
    """
    
    def __init__(self, model_dir: str = 'model'):
        """
        初始化分类器
        
        Args:
            model_dir: 模型文件目录
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_extractor = None
        
        # 录音参数
        self.sample_rate = 22050
        self.duration = 3.0  # 录音时长（秒）
        self.channels = 1    # 单声道
        
        # 加载模型和预处理器
        self.load_model()
    
    def load_model(self) -> None:
        """
        加载训练好的模型和预处理器
        """
        print("正在加载模型...")
        
        try:
            # 加载模型
            model_path = os.path.join(self.model_dir, 'frequency_classifier.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            self.model = joblib.load(model_path)
            
            # 加载标准化器
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # 加载标签编码器
            encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"标签编码器文件不存在: {encoder_path}")
            self.label_encoder = joblib.load(encoder_path)
            
            # 加载特征提取器配置
            config_path = os.path.join(self.model_dir, 'feature_config.pkl')
            if os.path.exists(config_path):
                config = joblib.load(config_path)
                self.feature_extractor = AudioFeatureExtractor(
                    sample_rate=config['sample_rate'],
                    n_mfcc=config['n_mfcc'],
                    n_fft=config['n_fft'],
                    hop_length=config['hop_length'],
                    duration=config['duration']
                )
                self.sample_rate = config['sample_rate']
                self.duration = config['duration']
            else:
                # 使用默认配置
                self.feature_extractor = AudioFeatureExtractor()
            
            print("模型加载成功！")
            print(f"支持的类别: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            sys.exit(1)
    
    def record_audio(self) -> np.ndarray:
        """
        录制音频
        
        Returns:
            录制的音频数据
        """
        try:
            print(f"开始录音 {self.duration} 秒...")
            
            # 录制音频
            audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            
            # 等待录音完成
            sd.wait()
            
            # 转换为一维数组
            audio_data = audio_data.flatten()
            
            print("录音完成")
            return audio_data
            
        except Exception as e:
            print(f"录音时出错: {e}")
            return np.array([])
    
    def classify_audio(self, audio_data: np.ndarray, include_additional_features: bool = True) -> tuple:
        """
        对音频数据进行分类
        
        Args:
            audio_data: 音频数据
            include_additional_features: 是否包含额外特征
            
        Returns:
            (预测类别, 置信度分数)
        """
        try:
            # 提取特征
            features = self.feature_extractor.extract_features_from_audio(
                audio_data, include_additional_features
            )
            
            # 重塑为二维数组
            features = features.reshape(1, -1)
            
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            # 预测
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # 解码标签
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            return predicted_class, confidence, probabilities
            
        except Exception as e:
            print(f"分类时出错: {e}")
            return "unknown", 0.0, np.array([])
    
    def get_audio_info(self, audio_data: np.ndarray) -> dict:
        """
        获取音频基本信息
        
        Args:
            audio_data: 音频数据
            
        Returns:
            音频信息字典
        """
        try:
            # 计算音频统计信息
            rms = np.sqrt(np.mean(audio_data**2))
            max_amplitude = np.max(np.abs(audio_data))
            
            # 简单的音量检测
            volume_level = "静音" if rms < 0.01 else "正常" if rms < 0.1 else "较大"
            
            return {
                'duration': len(audio_data) / self.sample_rate,
                'rms': rms,
                'max_amplitude': max_amplitude,
                'volume_level': volume_level
            }
            
        except Exception as e:
            print(f"获取音频信息时出错: {e}")
            return {}
    
    def print_classification_result(self, predicted_class: str, confidence: float, 
                                  probabilities: np.ndarray, audio_info: dict) -> None:
        """
        打印分类结果
        
        Args:
            predicted_class: 预测类别
            confidence: 置信度
            probabilities: 各类别概率
            audio_info: 音频信息
        """
        print("\n" + "="*50)
        print("分类结果")
        print("="*50)
        
        # 音频信息
        if audio_info:
            print(f"音频时长: {audio_info.get('duration', 0):.2f} 秒")
            print(f"音量水平: {audio_info.get('volume_level', '未知')}")
            print(f"RMS值: {audio_info.get('rms', 0):.4f}")
        
        # 分类结果
        print(f"\n预测类别: {predicted_class}")
        print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # 各类别概率
        if len(probabilities) > 0:
            print("\n各类别概率:")
            for i, class_name in enumerate(self.label_encoder.classes_):
                prob = probabilities[i]
                bar_length = int(prob * 20)  # 20字符长度的进度条
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {class_name:10s}: {prob:.4f} |{bar}| {prob*100:.1f}%")
        
        print("="*50)
    
    def run_continuous_inference(self, include_additional_features: bool = True) -> None:
        """
        运行连续推理
        
        Args:
            include_additional_features: 是否包含额外特征
        """
        print("\n开始实时音频分类...")
        print("按 Ctrl+C 退出")
        print("\n配置信息:")
        print(f"  采样率: {self.sample_rate} Hz")
        print(f"  录音时长: {self.duration} 秒")
        print(f"  包含额外特征: {include_additional_features}")
        
        try:
            while True:
                # 录制音频
                audio_data = self.record_audio()
                
                if len(audio_data) == 0:
                    print("录音失败，跳过此次分类")
                    continue
                
                # 获取音频信息
                audio_info = self.get_audio_info(audio_data)
                
                # 检查音频是否太安静
                if audio_info.get('rms', 0) < 0.005:
                    print("检测到静音，跳过分类")
                    time.sleep(1)
                    continue
                
                # 分类
                predicted_class, confidence, probabilities = self.classify_audio(
                    audio_data, include_additional_features
                )
                
                # 打印结果
                self.print_classification_result(
                    predicted_class, confidence, probabilities, audio_info
                )
                
                # 等待一段时间再进行下次录音
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n程序已停止")
        except Exception as e:
            print(f"\n运行时出错: {e}")
    
    def test_single_classification(self, include_additional_features: bool = True) -> None:
        """
        测试单次分类
        
        Args:
            include_additional_features: 是否包含额外特征
        """
        print("\n单次音频分类测试")
        print("准备录音...")
        
        try:
            # 录制音频
            audio_data = self.record_audio()
            
            if len(audio_data) == 0:
                print("录音失败")
                return
            
            # 获取音频信息
            audio_info = self.get_audio_info(audio_data)
            
            # 分类
            predicted_class, confidence, probabilities = self.classify_audio(
                audio_data, include_additional_features
            )
            
            # 打印结果
            self.print_classification_result(
                predicted_class, confidence, probabilities, audio_info
            )
            
        except Exception as e:
            print(f"测试时出错: {e}")


def check_audio_devices():
    """
    检查可用的音频设备
    """
    print("可用的音频设备:")
    print(sd.query_devices())
    print()


def main():
    """
    主函数
    """
    print("=" * 60)
    print("树莓派实时音频频率分类系统")
    print("=" * 60)
    
    # 检查模型目录
    model_dir = 'model'
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录 {model_dir} 不存在")
        print("请先运行 train_model.py 训练模型")
        return
    
    # 检查音频设备
    try:
        check_audio_devices()
    except Exception as e:
        print(f"检查音频设备时出错: {e}")
    
    # 创建分类器
    try:
        classifier = RealTimeFrequencyClassifier(model_dir=model_dir)
    except Exception as e:
        print(f"初始化分类器失败: {e}")
        return
    
    # 选择运行模式
    print("请选择运行模式:")
    print("1. 单次测试")
    print("2. 连续推理")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        include_additional_features = True  # 是否包含额外特征
        
        if choice == '1':
            classifier.test_single_classification(include_additional_features)
        elif choice == '2':
            classifier.run_continuous_inference(include_additional_features)
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"运行时出错: {e}")


if __name__ == '__main__':
    main()