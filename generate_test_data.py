#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据生成脚本

功能：
1. 生成不同频率的正弦波音频文件
2. 生成混合频率的复合音频
3. 添加噪声和变化
4. 自动组织到对应的类别文件夹

使用方法：
    python generate_test_data.py
"""

import os
import time
import numpy as np
import soundfile as sf
from tqdm import tqdm
import random
import psutil
from config_manager import config_manager
from logger_setup import LoggerSetup, PerformanceLogger, setup_logger
from performance_monitor import monitor_performance, PerformanceMonitor


class TestDataGenerator:
    """
    测试数据生成器
    """
    
    def __init__(self, data_dir: str = None, sample_rate: int = None, duration: float = None):
        """
        初始化生成器
        
        Args:
            data_dir: 数据保存目录
            sample_rate: 采样率
            duration: 音频时长（秒）
        """
        # 从配置文件获取默认值
        self.data_dir = data_dir or config_manager.get('data.data_dir', 'data')
        self.sample_rate = sample_rate or config_manager.get('audio.sample_rate', 22050)
        self.duration = duration or config_manager.get('audio.duration', 3.0)
        self.num_samples = int(self.sample_rate * self.duration)
        
        # 初始化日志器
        self.logger = LoggerSetup.get_logger('data_generator')
        self.performance_logger = PerformanceLogger()
        
        # 记录内存使用情况
        memory_monitor = PerformanceMonitor()
        memory_info = memory_monitor.get_memory_usage()
        self.logger.info(f"初始化数据生成器 - 内存使用: {memory_info['memory_percent']:.1f}%")
        
        self.logger.info(f"数据生成器配置 - 目录: {self.data_dir}, 采样率: {self.sample_rate}, 时长: {self.duration}s")
        
        # 从配置文件获取类别信息
        self.categories = config_manager.get('frequency_categories', {
            'low_freq': {'name': 'low_freq'},
            'mid_freq': {'name': 'mid_freq'},
            'high_freq': {'name': 'high_freq'}
        })
        
        # 确保目录存在
        for category in self.categories.keys():
            category_path = os.path.join(self.data_dir, category)
            os.makedirs(category_path, exist_ok=True)
            self.logger.debug(f"创建目录: {category_path}")
    
    def generate_sine_wave(self, frequency: float, amplitude: float = 0.5, phase: float = 0) -> np.ndarray:
        """
        生成正弦波
        
        Args:
            frequency: 频率 (Hz)
            amplitude: 振幅
            phase: 相位
            
        Returns:
            音频数据数组
        """
        t = np.linspace(0, self.duration, self.num_samples)
        wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return wave
    
    def generate_complex_wave(self, frequencies: list, amplitudes: list = None) -> np.ndarray:
        """
        生成复合波（多个频率叠加）
        
        Args:
            frequencies: 频率列表
            amplitudes: 振幅列表
            
        Returns:
            音频数据数组
        """
        if amplitudes is None:
            amplitudes = [1.0 / len(frequencies)] * len(frequencies)
        
        wave = np.zeros(self.num_samples)
        for freq, amp in zip(frequencies, amplitudes):
            wave += self.generate_sine_wave(freq, amp)
        
        # 归一化
        wave = wave / np.max(np.abs(wave)) * 0.7
        return wave
    
    def add_noise(self, wave: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        添加噪声
        
        Args:
            wave: 原始音频
            noise_level: 噪声水平
            
        Returns:
            添加噪声后的音频
        """
        noise = np.random.normal(0, noise_level, len(wave))
        return wave + noise
    
    def apply_envelope(self, wave: np.ndarray, attack: float = 0.1, decay: float = 0.1) -> np.ndarray:
        """
        应用包络（淡入淡出）
        
        Args:
            wave: 原始音频
            attack: 淡入时间（秒）
            decay: 淡出时间（秒）
            
        Returns:
            应用包络后的音频
        """
        envelope = np.ones(len(wave))
        
        # 淡入
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # 淡出
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        return wave * envelope
    
    def generate_frequency_sweep(self, start_freq: float, end_freq: float) -> np.ndarray:
        """
        生成频率扫描信号
        
        Args:
            start_freq: 起始频率
            end_freq: 结束频率
            
        Returns:
            音频数据数组
        """
        t = np.linspace(0, self.duration, self.num_samples)
        # 线性频率变化
        freq_t = start_freq + (end_freq - start_freq) * t / self.duration
        # 计算相位
        phase = 2 * np.pi * np.cumsum(freq_t) / self.sample_rate
        wave = 0.5 * np.sin(phase)
        return wave
    
    @monitor_performance
    def generate_low_freq_samples(self, num_samples: int = None) -> None:
        """
        生成低频样本
        
        Args:
            num_samples: 样本数量
        """
        # 从配置文件获取样本数量
        if num_samples is None:
            num_samples = config_manager.get('test_data_generation.samples_per_category', 20)
        
        self.logger.info(f"开始生成低频样本 - 数量: {num_samples}")
        print("生成低频样本...")
        
        # 从配置文件获取频率范围
        low_freq_config = config_manager.get('frequency_categories.low_freq', {})
        frequencies = low_freq_config.get('test_frequencies', [100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900])
        self.logger.debug(f"低频测试频率: {frequencies}")
        
        generated_files = []
        
        for i in tqdm(range(num_samples), desc="低频样本"):
            # 选择主频率
            main_freq = random.choice(frequencies)
            
            if i % 4 == 0:
                # 纯正弦波
                wave = self.generate_sine_wave(main_freq)
                filename = f"sine_{main_freq}hz_{i}.wav"
            elif i % 4 == 1:
                # 复合波（基频 + 谐波）
                harmonics = [main_freq, main_freq * 2, main_freq * 3]
                amplitudes = [0.7, 0.2, 0.1]
                wave = self.generate_complex_wave(harmonics, amplitudes)
                filename = f"complex_{main_freq}hz_{i}.wav"
            elif i % 4 == 2:
                # 频率扫描
                end_freq = min(main_freq + 200, 950)
                wave = self.generate_frequency_sweep(main_freq, end_freq)
                filename = f"sweep_{main_freq}to{end_freq}hz_{i}.wav"
            else:
                # 带噪声的正弦波
                wave = self.generate_sine_wave(main_freq)
                wave = self.add_noise(wave, 0.03)
                filename = f"noisy_{main_freq}hz_{i}.wav"
            
            # 应用包络
            wave = self.apply_envelope(wave)
            
            # 保存文件
            filepath = os.path.join(self.data_dir, 'low_freq', filename)
            sf.write(filepath, wave, self.sample_rate)
            generated_files.append(filename)
            self.logger.debug(f"保存低频样本: {filename}")
        
        # 记录生成结果
        self.logger.info(f"低频样本生成完成 - 共生成 {len(generated_files)} 个文件")
        self.performance_logger.log_performance({
            'operation': 'generate_low_freq_samples',
            'samples_generated': len(generated_files),
            'frequency_range': f"{min(frequencies)}-{max(frequencies)}Hz",
            'sample_rate': self.sample_rate,
            'duration': self.duration
        })
    
    @monitor_performance
    def generate_mid_freq_samples(self, num_samples: int = None) -> None:
        """
        生成中频样本
        
        Args:
            num_samples: 样本数量
        """
        # 从配置文件获取样本数量
        if num_samples is None:
            num_samples = config_manager.get('test_data_generation.samples_per_category', 20)
        
        self.logger.info(f"开始生成中频样本 - 数量: {num_samples}")
        print("生成中频样本...")
        
        # 从配置文件获取频率范围
        mid_freq_config = config_manager.get('frequency_categories.mid_freq', {})
        frequencies = mid_freq_config.get('test_frequencies', [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200, 3500, 3800])
        self.logger.debug(f"中频测试频率: {frequencies}")
        
        generated_files = []
        
        for i in tqdm(range(num_samples), desc="中频样本"):
            # 选择主频率
            main_freq = random.choice(frequencies)
            
            if i % 5 == 0:
                # 纯正弦波
                wave = self.generate_sine_wave(main_freq)
                filename = f"sine_{main_freq}hz_{i}.wav"
            elif i % 5 == 1:
                # 复合波
                harmonics = [main_freq, main_freq * 1.5, main_freq * 2]
                amplitudes = [0.6, 0.3, 0.1]
                wave = self.generate_complex_wave(harmonics, amplitudes)
                filename = f"complex_{main_freq}hz_{i}.wav"
            elif i % 5 == 2:
                # 调制波
                carrier = self.generate_sine_wave(main_freq, 0.5)
                modulator = self.generate_sine_wave(main_freq * 0.1, 0.3)
                wave = carrier * (1 + modulator)
                filename = f"modulated_{main_freq}hz_{i}.wav"
            elif i % 5 == 3:
                # 频率扫描
                start_freq = max(main_freq - 300, 1000)
                end_freq = min(main_freq + 300, 3900)
                wave = self.generate_frequency_sweep(start_freq, end_freq)
                filename = f"sweep_{start_freq}to{end_freq}hz_{i}.wav"
            else:
                # 多音调混合
                freqs = [main_freq, main_freq + 200, main_freq + 400]
                wave = self.generate_complex_wave(freqs)
                wave = self.add_noise(wave, 0.02)
                filename = f"multitone_{main_freq}hz_{i}.wav"
            
            # 应用包络
            wave = self.apply_envelope(wave)
            
            # 保存文件
            filepath = os.path.join(self.data_dir, 'high_freq', filename)
            sf.write(filepath, wave, self.sample_rate)
            generated_files.append(filename)
            self.logger.debug(f"保存高频样本: {filename}")
        
        # 记录生成结果
        self.logger.info(f"高频样本生成完成 - 共生成 {len(generated_files)} 个文件")
        self.performance_logger.log_performance({
             'operation': 'generate_high_freq_samples',
             'samples_generated': len(generated_files),
             'frequency_range': f"{min(frequencies)}-{max(frequencies)}Hz",
             'sample_rate': self.sample_rate,
             'duration': self.duration
         })
    
    @monitor_performance
    def generate_high_freq_samples(self, num_samples: int = None) -> None:
        """
        生成高频样本
        
        Args:
            num_samples: 样本数量
        """
        # 从配置文件获取样本数量
        if num_samples is None:
            num_samples = config_manager.get('test_data_generation.samples_per_category', 20)
        
        self.logger.info(f"开始生成高频样本 - 数量: {num_samples}")
        print("生成高频样本...")
        
        # 从配置文件获取频率范围
        high_freq_config = config_manager.get('frequency_categories.high_freq', {})
        frequencies = high_freq_config.get('test_frequencies', [4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])
        self.logger.debug(f"高频测试频率: {frequencies}")
        
        generated_files = []
        
        for i in tqdm(range(num_samples), desc="高频样本"):
            # 选择主频率
            main_freq = random.choice(frequencies)
            
            if i % 4 == 0:
                # 纯正弦波
                wave = self.generate_sine_wave(main_freq)
                filename = f"sine_{main_freq}hz_{i}.wav"
            elif i % 4 == 1:
                # 脉冲波
                wave = self.generate_sine_wave(main_freq)
                # 创建脉冲包络
                pulse_freq = 10  # 10Hz脉冲
                pulse_envelope = (np.sin(2 * np.pi * pulse_freq * np.linspace(0, self.duration, self.num_samples)) + 1) / 2
                wave = wave * pulse_envelope
                filename = f"pulse_{main_freq}hz_{i}.wav"
            elif i % 4 == 2:
                # 高频复合波
                harmonics = [main_freq, main_freq * 0.8, main_freq * 1.2]
                amplitudes = [0.5, 0.3, 0.2]
                wave = self.generate_complex_wave(harmonics, amplitudes)
                filename = f"complex_{main_freq}hz_{i}.wav"
            else:
                # 带噪声的高频
                wave = self.generate_sine_wave(main_freq, 0.4)
                wave = self.add_noise(wave, 0.05)
                filename = f"noisy_{main_freq}hz_{i}.wav"
            
            # 应用包络
            wave = self.apply_envelope(wave, attack=0.05, decay=0.05)
            
            # 保存文件
            filepath = os.path.join(self.data_dir, 'high_freq', filename)
            sf.write(filepath, wave, self.sample_rate)
    
    @monitor_performance
    def generate_all_samples(self, samples_per_category: int = None) -> None:
        """
        生成所有类别的样本
        
        Args:
            samples_per_category: 每个类别的样本数量
        """
        # 从配置文件获取样本数量
        if samples_per_category is None:
            samples_per_category = config_manager.get('test_data_generation.samples_per_category', 20)
        
        self.logger.info(f"开始生成所有类别的测试数据 - 每个类别 {samples_per_category} 个样本")
        print(f"开始生成测试数据，每个类别 {samples_per_category} 个样本")
        print(f"采样率: {self.sample_rate} Hz")
        print(f"音频时长: {self.duration} 秒")
        print()
        
        # 生成各类别样本
        self.generate_low_freq_samples(samples_per_category)
        self.generate_mid_freq_samples(samples_per_category)
        self.generate_high_freq_samples(samples_per_category)
        
        print("\n测试数据生成完成！")
        print(f"数据保存在: {self.data_dir}")
        
        # 统计生成的文件
        total_files = 0
        category_stats = {}
        for category in ['low_freq', 'mid_freq', 'high_freq']:
            category_path = os.path.join(self.data_dir, category)
            files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            file_count = len(files)
            total_files += file_count
            category_stats[category] = file_count
            print(f"  {category}: {file_count} 个文件")
            self.logger.debug(f"{category}目录: {file_count} 个文件")
        
        self.logger.info(f"测试数据生成完成 - 总共生成 {total_files} 个文件")
        
        # 记录性能信息
        self.performance_logger.log_performance({
            'operation': 'generate_all_samples',
            'total_files_generated': total_files,
            'samples_per_category': samples_per_category,
            'categories': category_stats,
            'sample_rate': self.sample_rate,
            'duration': self.duration
        })


def main():
    """
    主函数
    """
    # 初始化日志系统
    logger = setup_logger('test_data_generator')
    performance_logger = PerformanceLogger('test_data_generation')
    
    logger.info("启动音频测试数据生成器")
    logger.info(f"内存使用情况: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    
    print("=" * 60)
    print("音频测试数据生成器")
    print("=" * 60)
    
    # 从配置文件获取参数
    data_dir = config_manager.get('paths.data_dir', 'data')
    samples_per_category = config_manager.get('test_data_generation.samples_per_category', 25)
    sample_rate = config_manager.get('audio.sample_rate', 22050)
    duration = config_manager.get('audio.duration', 3.0)
    
    logger.info(f"配置参数 - 数据目录: {data_dir}, 每类样本数: {samples_per_category}, 采样率: {sample_rate}, 时长: {duration}s")
    
    # 检查是否覆盖现有数据
    existing_files = []
    for category in ['low_freq', 'mid_freq', 'high_freq']:
        category_path = os.path.join(data_dir, category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            existing_files.extend(files)
            logger.debug(f"发现 {category} 目录中有 {len(files)} 个现有文件")
    
    if existing_files:
        logger.warning(f"发现 {len(existing_files)} 个现有音频文件")
        print(f"警告: 发现 {len(existing_files)} 个现有音频文件")
        response = input("是否继续生成新数据？这将添加到现有数据中 (y/n): ")
        if response.lower() != 'y':
            logger.info("用户取消操作")
            print("操作已取消")
            return
        else:
            logger.info("用户选择继续生成数据")
    else:
        logger.info("未发现现有数据文件，开始生成新数据")
    
    # 创建生成器
    logger.info("初始化测试数据生成器")
    try:
        generator = TestDataGenerator(
            data_dir=data_dir,
            sample_rate=sample_rate,
            duration=duration
        )
        logger.info("测试数据生成器初始化成功")
    except Exception as e:
        logger.error(f"生成器初始化失败: {e}")
        print(f"生成器初始化失败: {e}")
        return
    
    # 生成数据
    start_time = time.time()
    try:
        logger.info(f"开始生成测试数据 - 每类 {samples_per_category} 个样本")
        generator.generate_all_samples(samples_per_category)
        
        generation_time = time.time() - start_time
        logger.info(f"数据生成完成，耗时: {generation_time:.2f}秒")
        
        print("\n生成完成！现在可以运行以下命令训练模型:")
        print("python train_model.py")
        
        # 记录会话性能
        performance_logger.log_performance({
            'operation': 'main_session',
            'total_generation_time': generation_time,
            'samples_per_category': samples_per_category,
            'total_categories': 3,
            'sample_rate': sample_rate,
            'duration': duration,
            'status': 'completed'
        })
        
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        print("\n操作已中断")
        performance_logger.log_performance({
            'operation': 'main_session',
            'status': 'interrupted_by_user',
            'samples_per_category': samples_per_category
        })
    except Exception as e:
        logger.error(f"生成过程中出现错误: {e}", exc_info=True)
        print(f"\n生成过程中出现错误: {e}")
        performance_logger.log_performance({
            'operation': 'main_session',
            'status': 'error',
            'error_message': str(e),
            'samples_per_category': samples_per_category
        })
    finally:
        logger.info("测试数据生成器会话结束")


if __name__ == '__main__':
    main()