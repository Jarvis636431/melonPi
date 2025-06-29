# 音频频率分类项目

基于机器学习的音频频率分类系统，支持在PC上训练模型，在树莓派上进行实时推理。

## 项目概述

本项目实现了一个完整的音频频率分类系统，能够：
- 根据音频样本的主频特征进行分类（低频、中频、高频）
- 使用MFCC特征提取和随机森林分类器
- 支持PC端模型训练和树莓派端实时推理

## 项目结构

```
frequency_classifier_project/
├── data/                      # 音频样本数据
│   ├── low_freq/             # 低频音频文件 (<1000Hz)
│   ├── mid_freq/             # 中频音频文件 (1000-4000Hz)
│   └── high_freq/            # 高频音频文件 (>4000Hz)
├── model/                     # 训练后的模型文件
│   ├── frequency_classifier.pkl    # 训练好的分类器
│   ├── scaler.pkl                 # 特征标准化器
│   ├── label_encoder.pkl          # 标签编码器
│   └── feature_config.pkl         # 特征提取配置
├── train_model.py             # 模型训练脚本
├── inference_pi.py            # 树莓派推理脚本
├── utils.py                   # 特征提取和工具函数
├── requirements.txt           # Python依赖包
└── README.md                  # 项目说明文档
```

## 环境要求

### Python版本
- Python 3.7+

### 系统依赖

**Ubuntu/Debian (包括树莓派OS):**
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-dev
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install libsndfile1-dev
```

**macOS:**
```bash
brew install portaudio
brew install libsndfile
```

**Windows:**
- 安装 Microsoft Visual C++ Build Tools
- 或使用 conda 环境

## 安装步骤

1. **克隆项目**
```bash
git clone <your-repo-url>
cd frequency_classifier_project
```

2. **创建虚拟环境（推荐）**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

## 使用指南

### 1. 准备训练数据

在 `data/` 目录下按类别组织音频文件：

```
data/
├── low_freq/
│   ├── low_sample_1.wav
│   ├── low_sample_2.wav
│   └── ...
├── mid_freq/
│   ├── mid_sample_1.wav
│   ├── mid_sample_2.wav
│   └── ...
└── high_freq/
    ├── high_sample_1.wav
    ├── high_sample_2.wav
    └── ...
```

**支持的音频格式：**
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- M4A (.m4a)

**音频要求：**
- 建议每个类别至少20个样本
- 音频时长建议3-10秒
- 采样率不限（会自动重采样到22050Hz）

### 2. 训练模型

在PC上运行训练脚本：

```bash
python train_model.py
```

**训练过程包括：**
- 自动加载和预处理音频数据
- 提取MFCC特征（13维均值 + 13维标准差 + 5维额外特征）
- 超参数优化（网格搜索）
- 模型评估（交叉验证、混淆矩阵、特征重要性）
- 保存训练好的模型

**输出文件：**
- `model/frequency_classifier.pkl` - 训练好的分类器
- `model/scaler.pkl` - 特征标准化器
- `model/label_encoder.pkl` - 标签编码器
- `model/feature_config.pkl` - 特征提取配置
- `model/confusion_matrix.png` - 混淆矩阵图
- `model/feature_importance.png` - 特征重要性图

### 3. 树莓派部署

**3.1 传输文件到树莓派**
```bash
# 复制整个项目到树莓派
scp -r frequency_classifier_project/ pi@<raspberry-pi-ip>:~/
```

**3.2 在树莓派上安装依赖**
```bash
ssh pi@<raspberry-pi-ip>
cd ~/frequency_classifier_project
pip3 install -r requirements.txt
```

**3.3 运行推理脚本**
```bash
python3 inference_pi.py
```

**推理模式：**
1. **单次测试** - 录制一次音频并分类
2. **连续推理** - 持续录制和分类音频

### 4. 使用示例

**训练输出示例：**
```
正在加载数据集...
处理类别: low_freq
处理 low_freq: 100%|██████████| 25/25 [00:15<00:00,  1.67it/s]
处理类别: mid_freq
处理 mid_freq: 100%|██████████| 30/30 [00:18<00:00,  1.65it/s]
处理类别: high_freq
处理 high_freq: 100%|██████████| 28/28 [00:17<00:00,  1.64it/s]

数据集加载完成:
特征矩阵形状: (83, 31)
样本数量: 83
特征维度: 31

各类别样本数量:
  high_freq: 28
  low_freq: 25
  mid_freq: 30

测试集准确率: 0.9412
5折交叉验证平均准确率: 0.9277 (+/- 0.0849)
```

**推理输出示例：**
```
==================================================
分类结果
==================================================
音频时长: 3.00 秒
音量水平: 正常
RMS值: 0.0234

预测类别: mid_freq
置信度: 0.8500 (85.00%)

各类别概率:
  high_freq : 0.1000 |██░░░░░░░░░░░░░░░░░░| 10.0%
  low_freq  : 0.0500 |█░░░░░░░░░░░░░░░░░░░| 5.0%
  mid_freq  : 0.8500 |█████████████████░░░| 85.0%
==================================================
```

## 特征说明

### MFCC特征（26维）
- **MFCC 1-13 均值** - 梅尔频率倒谱系数的时间均值
- **MFCC 1-13 标准差** - 梅尔频率倒谱系数的时间标准差

### 额外特征（5维）
- **零交叉率 (ZCR)** - 信号穿越零点的频率
- **频谱质心** - 频谱的重心位置
- **频谱带宽** - 频谱的带宽
- **频谱滚降** - 85%能量所在的频率
- **RMS能量** - 均方根能量

## 配置参数

### 音频处理参数
```python
sample_rate = 22050    # 采样率
duration = 3.0         # 音频片段时长（秒）
n_mfcc = 13           # MFCC特征维数
n_fft = 2048          # FFT窗口大小
hop_length = 512      # 跳跃长度
```

### 模型参数
```python
# RandomForest默认参数
n_estimators = 100
max_depth = 20
min_samples_split = 5
min_samples_leaf = 2
max_features = 'sqrt'
```

## 性能优化

### 提高分类准确率
1. **增加训练数据** - 每个类别建议50+样本
2. **数据质量** - 确保音频清晰，噪声较少
3. **特征工程** - 可以添加更多音频特征
4. **模型调优** - 调整超参数或尝试其他算法

### 树莓派性能优化
1. **减少特征维度** - 设置 `include_additional_features=False`
2. **调整录音参数** - 减少 `duration` 或降低 `sample_rate`
3. **模型简化** - 减少 `n_estimators` 或 `max_depth`

## 故障排除

### 常见问题

**1. 音频设备问题**
```bash
# 检查音频设备
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# 测试录音
python3 -c "import sounddevice as sd; import numpy as np; print('Recording...'); data = sd.rec(44100, samplerate=44100, channels=1); sd.wait(); print('Done')"
```

**2. 依赖安装问题**
```bash
# 如果librosa安装失败
pip install librosa --no-cache-dir

# 如果sounddevice安装失败
sudo apt-get install portaudio19-dev
pip install sounddevice
```

**3. 模型加载问题**
- 确保模型文件存在于 `model/` 目录
- 检查文件权限
- 确保Python版本兼容

**4. 分类效果不佳**
- 检查训练数据质量和数量
- 确保各类别数据平衡
- 调整特征提取参数
- 增加训练数据或改进数据预处理

### 调试模式

在脚本中添加调试信息：
```python
# 在 train_model.py 或 inference_pi.py 开头添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 添加新的频率类别
1. 在 `data/` 目录下创建新的类别文件夹
2. 添加对应的音频样本
3. 重新训练模型

### 集成到其他系统
```python
from utils import AudioFeatureExtractor
import joblib

# 加载模型
model = joblib.load('model/frequency_classifier.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

# 创建特征提取器
extractor = AudioFeatureExtractor()

# 分类音频文件
features = extractor.extract_features_from_file('test_audio.wav')
features_scaled = scaler.transform(features.reshape(1, -1))
prediction = model.predict(features_scaled)[0]
result = label_encoder.inverse_transform([prediction])[0]
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]