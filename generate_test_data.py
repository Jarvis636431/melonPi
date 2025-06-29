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
import numpy as np
import soundfile as sf
from tqdm import tqdm
import random


class TestDataGenerator:
    """
    测试数据生成器
    """
    
    def __init__(self, data_dir: str = 'data', sample_rate: int = 22050, duration: float = 3.0):
        """
        初始化生成器
        
        Args:
            data_dir: 数据保存目录
            sample_rate: 采样率
            duration: 音频时长（秒）
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        
        # 确保目录存在
        for category in ['low_freq', 'mid_freq', 'high_freq']:
            os.makedirs(os.path.join(data_dir, category), exist_ok=True)
    
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
    
    def generate_low_freq_samples(self, num_samples: int = 20) -> None:
        """
        生成低频样本
        
        Args:
            num_samples: 样本数量
        """
        print("生成低频样本...")
        
        frequencies = [100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900]
        
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
    
    def generate_mid_freq_samples(self, num_samples: int = 20) -> None:
        """
        生成中频样本
        
        Args:
            num_samples: 样本数量
        """
        print("生成中频样本...")
        
        frequencies = [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200, 3500, 3800]
        
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
            filepath = os.path.join(self.data_dir, 'mid_freq', filename)
            sf.write(filepath, wave, self.sample_rate)
    
    def generate_high_freq_samples(self, num_samples: int = 20) -> None:
        """
        生成高频样本
        
        Args:
            num_samples: 样本数量
        """
        print("生成高频样本...")
        
        frequencies = [4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
        
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
    
    def generate_all_samples(self, samples_per_category: int = 20) -> None:
        """
        生成所有类别的样本
        
        Args:
            samples_per_category: 每个类别的样本数量
        """
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
        for category in ['low_freq', 'mid_freq', 'high_freq']:
            category_path = os.path.join(self.data_dir, category)
            files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            print(f"  {category}: {len(files)} 个文件")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("音频测试数据生成器")
    print("=" * 60)
    
    # 配置参数
    data_dir = 'data'
    samples_per_category = 25  # 每个类别生成25个样本
    sample_rate = 22050
    duration = 3.0
    
    # 检查是否覆盖现有数据
    existing_files = []
    for category in ['low_freq', 'mid_freq', 'high_freq']:
        category_path = os.path.join(data_dir, category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            existing_files.extend(files)
    
    if existing_files:
        print(f"警告: 发现 {len(existing_files)} 个现有音频文件")
        response = input("是否继续生成新数据？这将添加到现有数据中 (y/n): ")
        if response.lower() != 'y':
            print("操作已取消")
            return
    
    # 创建生成器
    generator = TestDataGenerator(
        data_dir=data_dir,
        sample_rate=sample_rate,
        duration=duration
    )
    
    # 生成数据
    try:
        generator.generate_all_samples(samples_per_category)
        
        print("\n生成完成！现在可以运行以下命令训练模型:")
        print("python train_model.py")
        
    except KeyboardInterrupt:
        print("\n操作已中断")
    except Exception as e:
        print(f"\n生成过程中出现错误: {e}")


if __name__ == '__main__':
    main()