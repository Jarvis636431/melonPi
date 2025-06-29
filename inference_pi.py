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
from config_manager import config_manager
from logger_setup import LoggerSetup, PerformanceLogger
from performance_monitor import monitor_performance, PerformanceMonitor
import warnings
warnings.filterwarnings('ignore')


class RealTimeFrequencyClassifier:
    """
    实时频率分类器
    """
    
    def __init__(self, model_dir: str = None):
        """
        初始化分类器
        
        Args:
            model_dir: 模型文件目录
        """
        # 从配置文件获取默认值
        self.model_dir = model_dir or config_manager.get('raspberry_pi.model_dir', 'model')
        
        # 初始化日志器
        self.logger = LoggerSetup.get_logger('inference')
        self.performance_logger = PerformanceLogger()
        
        # 记录内存使用情况
        memory_monitor = PerformanceMonitor()
        memory_info = memory_monitor.get_memory_usage()
        self.logger.info(f"初始化推理器 - 内存使用: {memory_info['memory_percent']:.1f}%")
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_extractor = None
        
        # 从配置文件获取录音参数
        audio_config = config_manager.get('audio', {})
        self.sample_rate = audio_config.get('sample_rate', 22050)
        self.duration = audio_config.get('duration', 3.0)
        self.channels = 1    # 单声道
        
        self.logger.info(f"推理器初始化完成 - 模型目录: {self.model_dir}")
        self.logger.info(f"音频配置 - 采样率: {self.sample_rate}, 时长: {self.duration}s")
        
        # 加载模型和预处理器
        self.load_model()
    
    @monitor_performance
    def load_model(self) -> None:
        """
        加载训练好的模型和预处理器
        """
        self.logger.info("开始加载模型和预处理器")
        
        try:
            # 加载模型
            model_path = os.path.join(self.model_dir, 'frequency_classifier.pkl')
            self.logger.debug(f"检查模型文件: {model_path}")
            if not os.path.exists(model_path):
                error_msg = f"模型文件不存在: {model_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            self.logger.info(f"加载模型文件: {model_path}")
            self.model = joblib.load(model_path)
            model_size = os.path.getsize(model_path) / 1024 / 1024  # MB
            self.logger.info(f"模型加载成功，文件大小: {model_size:.2f} MB")
            
            # 加载标准化器
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            self.logger.debug(f"检查标准化器文件: {scaler_path}")
            if not os.path.exists(scaler_path):
                error_msg = f"标准化器文件不存在: {scaler_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            self.logger.info(f"加载标准化器: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # 加载标签编码器
            encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            self.logger.debug(f"检查标签编码器文件: {encoder_path}")
            if not os.path.exists(encoder_path):
                error_msg = f"标签编码器文件不存在: {encoder_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            self.logger.info(f"加载标签编码器: {encoder_path}")
            self.label_encoder = joblib.load(encoder_path)
            self.logger.info(f"支持的类别数量: {len(self.label_encoder.classes_)}")
            self.logger.debug(f"支持的类别: {list(self.label_encoder.classes_)}")
            
            # 加载特征提取器配置
            config_path = os.path.join(self.model_dir, 'feature_config.pkl')
            self.logger.debug(f"检查特征配置文件: {config_path}")
            if os.path.exists(config_path):
                self.logger.info(f"加载特征配置: {config_path}")
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
                self.logger.info(f"使用保存的特征配置 - MFCC: {config['n_mfcc']}, 采样率: {config['sample_rate']}")
            else:
                # 使用默认配置
                self.logger.warning(f"特征配置文件不存在，使用默认配置")
                self.feature_extractor = AudioFeatureExtractor()
            
            # 记录性能信息
            self.performance_logger.log_performance({
                'operation': 'model_loading',
                'model_classes': len(self.label_encoder.classes_),
                'feature_config_exists': os.path.exists(config_path),
                'model_size_mb': model_size
            })
            
            self.logger.info("所有模型组件加载成功！")
            
        except Exception as e:
            error_msg = f"加载模型时出错: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)
    
    @monitor_performance
    def record_audio(self) -> np.ndarray:
        """
        录制音频
        
        Returns:
            录制的音频数据
        """
        try:
            self.logger.debug(f"开始录音 {self.duration} 秒，采样率: {self.sample_rate} Hz")
            
            # 录制音频
            samples = int(self.duration * self.sample_rate)
            self.logger.debug(f"录制样本数: {samples}")
            
            audio_data = sd.rec(
                samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            
            # 等待录音完成
            sd.wait()
            
            # 转换为一维数组
            audio_data = audio_data.flatten()
            
            # 记录音频统计信息
            rms = np.sqrt(np.mean(audio_data**2))
            max_amp = np.max(np.abs(audio_data))
            self.logger.debug(f"录音完成 - 样本数: {len(audio_data)}, RMS: {rms:.4f}, 最大振幅: {max_amp:.4f}")
            
            return audio_data
            
        except Exception as e:
            error_msg = f"录音时出错: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            return np.array([])
    
    @monitor_performance
    def classify_audio(self, audio_data: np.ndarray, include_additional_features: bool = True) -> tuple:
        """
        对音频数据进行分类
        
        Args:
            audio_data: 音频数据
            include_additional_features: 是否包含额外特征
            
        Returns:
            (预测类别, 置信度分数, 各类别概率)
        """
        try:
            self.logger.debug(f"开始音频分类 - 音频长度: {len(audio_data)}, 包含额外特征: {include_additional_features}")
            
            # 提取特征
            features = self.feature_extractor.extract_features_from_audio(
                audio_data, include_additional_features
            )
            self.logger.debug(f"特征提取完成 - 特征维度: {features.shape}")
            
            # 重塑为二维数组
            features = features.reshape(1, -1)
            self.logger.debug(f"特征重塑后维度: {features.shape}")
            
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            self.logger.debug(f"特征标准化完成 - 范围: [{features_scaled.min():.4f}, {features_scaled.max():.4f}]")
            
            # 预测
            self.logger.debug("开始模型预测")
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            self.logger.debug(f"预测完成 - 预测索引: {prediction}")
            
            # 解码标签
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            # 记录预测结果
            self.logger.info(f"分类结果: {predicted_class}, 置信度: {confidence:.4f}")
            self.logger.debug(f"各类别概率: {dict(zip(self.label_encoder.classes_, probabilities))}")
            
            # 记录性能信息
            self.performance_logger.log_performance({
                'operation': 'audio_classification',
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'feature_dimension': features.shape[1],
                'include_additional_features': include_additional_features
            })
            
            return predicted_class, confidence, probabilities
            
        except Exception as e:
            error_msg = f"分类时出错: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            return "unknown", 0.0, np.array([])
    
    @monitor_performance
    def get_audio_info(self, audio_data: np.ndarray) -> dict:
        """
        获取音频基本信息
        
        Args:
            audio_data: 音频数据
            
        Returns:
            音频信息字典
        """
        try:
            self.logger.debug(f"分析音频信息 - 样本数: {len(audio_data)}")
            
            # 计算音频统计信息
            rms = np.sqrt(np.mean(audio_data**2))
            max_amplitude = np.max(np.abs(audio_data))
            duration = len(audio_data) / self.sample_rate
            
            # 从配置文件获取音量阈值
            volume_thresholds = config_manager.get('raspberry_pi.volume_thresholds', {
                'silent': 0.01,
                'normal': 0.1
            })
            
            # 音量检测
            if rms < volume_thresholds['silent']:
                volume_level = "静音"
            elif rms < volume_thresholds['normal']:
                volume_level = "正常"
            else:
                volume_level = "较大"
            
            audio_info = {
                'duration': duration,
                'rms': rms,
                'max_amplitude': max_amplitude,
                'volume_level': volume_level
            }
            
            self.logger.debug(f"音频信息: 时长={duration:.2f}s, RMS={rms:.4f}, 音量={volume_level}")
            
            return audio_info
            
        except Exception as e:
            error_msg = f"获取音频信息时出错: {e}"
            self.logger.error(error_msg)
            print(error_msg)
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
    
    @monitor_performance
    def run_continuous_inference(self, include_additional_features: bool = True) -> None:
        """
        运行连续推理
        
        Args:
            include_additional_features: 是否包含额外特征
        """
        self.logger.info("开始连续音频分类推理")
        self.logger.info(f"配置 - 采样率: {self.sample_rate} Hz, 时长: {self.duration}s, 额外特征: {include_additional_features}")
        
        print("\n开始实时音频分类...")
        print("按 Ctrl+C 退出")
        print("\n配置信息:")
        print(f"  采样率: {self.sample_rate} Hz")
        print(f"  录音时长: {self.duration} 秒")
        print(f"  包含额外特征: {include_additional_features}")
        
        # 从配置文件获取静音阈值和等待时间
        silent_threshold = config_manager.get('raspberry_pi.silent_threshold', 0.005)
        sleep_interval = config_manager.get('raspberry_pi.sleep_interval', 1)
        
        classification_count = 0
        successful_classifications = 0
        
        try:
            while True:
                classification_count += 1
                self.logger.debug(f"开始第 {classification_count} 次分类循环")
                
                # 录制音频
                audio_data = self.record_audio()
                
                if len(audio_data) == 0:
                    self.logger.warning("录音失败，跳过此次分类")
                    print("录音失败，跳过此次分类")
                    continue
                
                # 获取音频信息
                audio_info = self.get_audio_info(audio_data)
                
                # 检查音频是否太安静
                if audio_info.get('rms', 0) < silent_threshold:
                    self.logger.debug(f"检测到静音 (RMS: {audio_info.get('rms', 0):.6f} < {silent_threshold})，跳过分类")
                    print("检测到静音，跳过分类")
                    time.sleep(sleep_interval)
                    continue
                
                # 分类
                predicted_class, confidence, probabilities = self.classify_audio(
                    audio_data, include_additional_features
                )
                
                if predicted_class != "unknown":
                    successful_classifications += 1
                
                # 打印结果
                self.print_classification_result(
                    predicted_class, confidence, probabilities, audio_info
                )
                
                # 等待一段时间再进行下次录音
                time.sleep(sleep_interval)
                
        except KeyboardInterrupt:
            self.logger.info(f"用户中断程序 - 总分类次数: {classification_count}, 成功次数: {successful_classifications}")
            # 记录会话性能信息
            self.performance_logger.log_performance({
                'operation': 'continuous_inference_session',
                'total_classifications': classification_count,
                'successful_classifications': successful_classifications,
                'success_rate': successful_classifications / max(classification_count, 1)
            })
            print("\n\n程序已停止")
        except Exception as e:
            error_msg = f"运行时出错: {e}"
            self.logger.error(error_msg)
            print(f"\n{error_msg}")
    
    @monitor_performance
    def test_single_classification(self, include_additional_features: bool = True) -> None:
        """
        测试单次分类
        
        Args:
            include_additional_features: 是否包含额外特征
        """
        self.logger.info(f"开始单次分类测试 - 额外特征: {include_additional_features}")
        
        print("\n单次音频分类测试")
        print("准备录音...")
        
        try:
            # 录制音频
            audio_data = self.record_audio()
            
            if len(audio_data) == 0:
                self.logger.error("录音失败")
                print("录音失败")
                return
            
            # 获取音频信息
            audio_info = self.get_audio_info(audio_data)
            
            # 分类
            predicted_class, confidence, probabilities = self.classify_audio(
                audio_data, include_additional_features
            )
            
            # 记录测试结果
            self.performance_logger.log_performance({
                'operation': 'single_classification_test',
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'audio_duration': audio_info.get('duration', 0),
                'audio_rms': audio_info.get('rms', 0),
                'include_additional_features': include_additional_features
            })
            
            # 打印结果
            self.print_classification_result(
                predicted_class, confidence, probabilities, audio_info
            )
            
            self.logger.info(f"单次分类测试完成 - 结果: {predicted_class}")
            
        except Exception as e:
            error_msg = f"测试时出错: {e}"
            self.logger.error(error_msg)
            print(error_msg)


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
    # 初始化日志器
    logger = LoggerSetup.get_logger('main')
    performance_logger = PerformanceLogger()
    
    logger.info("启动树莓派实时音频频率分类系统")
    
    print("=" * 60)
    print("树莓派实时音频频率分类系统")
    print("=" * 60)
    
    # 从配置文件获取模型目录
    model_dir = config_manager.get('raspberry_pi.model_dir', 'model')
    logger.info(f"使用模型目录: {model_dir}")
    
    # 检查模型目录
    if not os.path.exists(model_dir):
        error_msg = f"错误: 模型目录 {model_dir} 不存在"
        logger.error(error_msg)
        print(error_msg)
        print("请先运行 train_model.py 训练模型")
        return
    
    # 检查音频设备
    try:
        logger.info("检查音频设备")
        check_audio_devices()
    except Exception as e:
        error_msg = f"检查音频设备时出错: {e}"
        logger.warning(error_msg)
        print(error_msg)
    
    # 创建分类器
    try:
        logger.info("初始化分类器")
        classifier = RealTimeFrequencyClassifier(model_dir=model_dir)
    except Exception as e:
        error_msg = f"初始化分类器失败: {e}"
        logger.error(error_msg)
        print(error_msg)
        return
    
    # 从配置文件获取运行选项
    include_additional_features = config_manager.get('raspberry_pi.include_additional_features', True)
    logger.info(f"额外特征设置: {include_additional_features}")
    
    # 选择运行模式
    print("请选择运行模式:")
    print("1. 单次测试")
    print("2. 连续推理")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        logger.info(f"用户选择运行模式: {choice}")
        
        if choice == '1':
            logger.info("开始单次测试模式")
            classifier.test_single_classification(include_additional_features)
        elif choice == '2':
            logger.info("开始连续推理模式")
            classifier.run_continuous_inference(include_additional_features)
        else:
            logger.warning(f"无效选择: {choice}")
            print("无效选择")
        
        # 记录会话信息
        performance_logger.log_performance({
            'operation': 'main_session',
            'mode': choice,
            'include_additional_features': include_additional_features,
            'model_dir': model_dir
        })
            
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        print("\n程序已退出")
    except Exception as e:
        error_msg = f"运行时出错: {e}"
        logger.error(error_msg)
        print(error_msg)


if __name__ == '__main__':
    main()