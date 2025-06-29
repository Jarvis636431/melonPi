# 音频数据目录说明

本目录用于存放训练用的音频样本数据。请按照以下结构组织您的音频文件：

## 目录结构

```
data/
├── low_freq/     # 低频音频文件 (< 1000 Hz)
├── mid_freq/     # 中频音频文件 (1000-4000 Hz)
├── high_freq/    # 高频音频文件 (> 4000 Hz)
└── README.md     # 本说明文件
```

## 音频文件要求

### 支持的格式
- WAV (.wav) - 推荐
- MP3 (.mp3)
- FLAC (.flac)
- M4A (.m4a)

### 音频质量要求
- **采样率**: 不限（程序会自动重采样到22050Hz）
- **时长**: 建议3-10秒
- **声道**: 单声道或立体声均可（会转换为单声道）
- **音质**: 尽量清晰，噪声较少

### 数据量建议
- **最少**: 每个类别至少10个样本
- **推荐**: 每个类别20-50个样本
- **理想**: 每个类别50+个样本

## 频率分类标准

### low_freq (低频)
- **频率范围**: < 1000 Hz
- **典型声音**: 
  - 低音乐器（大提琴、低音提琴、低音鼓）
  - 男性低音说话
  - 引擎低频噪声
  - 低频正弦波

### mid_freq (中频)
- **频率范围**: 1000-4000 Hz
- **典型声音**:
  - 人声（大部分语音频率）
  - 钢琴中音区
  - 吉他中音
  - 中频正弦波

### high_freq (高频)
- **频率范围**: > 4000 Hz
- **典型声音**:
  - 高音乐器（小提琴高音、长笛、铃铛）
  - 鸟叫声
  - 哨声
  - 高频正弦波

## 数据收集建议

### 1. 录制音频
```bash
# 使用 arecord (Linux)
arecord -f cd -t wav -d 5 sample.wav

# 使用 sox
sox -d sample.wav trim 0 5

# 使用 ffmpeg
ffmpeg -f alsa -i default -t 5 sample.wav
```

### 2. 生成测试音频
可以使用Python生成不同频率的正弦波作为测试数据：

```python
import numpy as np
import soundfile as sf

def generate_sine_wave(frequency, duration=3, sample_rate=22050, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave

# 生成低频样本 (200-800 Hz)
for i, freq in enumerate([200, 300, 500, 800]):
    wave = generate_sine_wave(freq)
    sf.write(f'low_freq/sine_{freq}hz_{i}.wav', wave, 22050)

# 生成中频样本 (1000-3500 Hz)
for i, freq in enumerate([1000, 1500, 2000, 2500, 3000, 3500]):
    wave = generate_sine_wave(freq)
    sf.write(f'mid_freq/sine_{freq}hz_{i}.wav', wave, 22050)

# 生成高频样本 (4000-8000 Hz)
for i, freq in enumerate([4000, 5000, 6000, 7000, 8000]):
    wave = generate_sine_wave(freq)
    sf.write(f'high_freq/sine_{freq}hz_{i}.wav', wave, 22050)
```

### 3. 从现有音频提取
```python
import librosa
import soundfile as sf

# 加载音频文件
audio, sr = librosa.load('source_audio.wav', sr=22050)

# 分割成3秒片段
segment_length = 3 * sr
for i in range(0, len(audio) - segment_length, segment_length):
    segment = audio[i:i + segment_length]
    sf.write(f'segment_{i//segment_length}.wav', segment, sr)
```

## 数据质量检查

训练前建议检查数据质量：

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_audio_frequency(file_path):
    # 加载音频
    y, sr = librosa.load(file_path)
    
    # 计算频谱
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(fft), 1/sr)
    
    # 找到主频
    idx = np.argmax(magnitude[:len(magnitude)//2])
    dominant_freq = frequency[idx]
    
    print(f"文件: {file_path}")
    print(f"主频: {dominant_freq:.2f} Hz")
    
    return dominant_freq

# 检查所有音频文件的主频
import glob
for category in ['low_freq', 'mid_freq', 'high_freq']:
    files = glob.glob(f'{category}/*.wav')
    print(f"\n{category} 类别:")
    for file in files:
        analyze_audio_frequency(file)
```

## 注意事项

1. **文件命名**: 使用有意义的文件名，如 `piano_c4_440hz.wav`
2. **避免重复**: 确保同一音频不要重复放在不同类别中
3. **数据平衡**: 尽量保持各类别样本数量相近
4. **质量控制**: 定期检查音频文件是否损坏或格式不正确
5. **版权注意**: 确保使用的音频文件有合法使用权

## 示例数据集

如果您需要快速开始，可以考虑使用以下公开数据集：

- **FreeSound**: https://freesound.org/
- **Common Voice**: https://commonvoice.mozilla.org/
- **LibriSpeech**: http://www.openslr.org/12/
- **ESC-50**: https://github.com/karolpiczak/ESC-50

记住要根据项目需求调整和标注这些数据。