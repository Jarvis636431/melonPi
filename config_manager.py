#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

功能：
1. 加载YAML配置文件
2. 提供配置访问接口
3. 配置验证和默认值处理
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """
    配置管理器类
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            if config is None:
                raise ValueError("配置文件为空或格式错误")
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"YAML配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")
    
    def _validate_config(self) -> None:
        """
        验证配置文件的必要字段
        """
        required_sections = ['audio', 'model', 'data', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必要部分: {section}")
        
        # 验证音频配置
        audio_config = self.config['audio']
        required_audio_fields = ['sample_rate', 'duration', 'n_mfcc']
        for field in required_audio_fields:
            if field not in audio_config:
                raise ValueError(f"音频配置缺少字段: {field}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持嵌套键）
        
        Args:
            key_path: 配置键路径，如 'audio.sample_rate'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_audio_config(self) -> Dict[str, Any]:
        """
        获取音频配置
        
        Returns:
            音频配置字典
        """
        return self.config.get('audio', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        Returns:
            模型配置字典
        """
        return self.config.get('model', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        获取数据配置
        
        Returns:
            数据配置字典
        """
        return self.config.get('data', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置
        
        Returns:
            日志配置字典
        """
        return self.config.get('logging', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """
        获取性能监控配置
        
        Returns:
            性能监控配置字典
        """
        return self.config.get('performance', {})
    
    def get_raspberry_pi_config(self) -> Dict[str, Any]:
        """
        获取树莓派配置
        
        Returns:
            树莓派配置字典
        """
        return self.config.get('raspberry_pi', {})
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        更新配置值
        
        Args:
            key_path: 配置键路径
            value: 新值
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # 导航到目标位置
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # 设置值
        config_ref[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            output_path: 输出文件路径，默认为原配置文件路径
        """
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            raise RuntimeError(f"保存配置文件失败: {e}")
    
    def create_directories(self) -> None:
        """
        根据配置创建必要的目录
        """
        directories = [
            self.get('data.data_dir', 'data'),
            self.get('data.model_dir', 'model'),
            'logs',
            self.get('visualization.plot_dir', 'plots')
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logging.info(f"创建目录: {directory}")


# 全局配置实例
_config_manager = None


def get_config_manager(config_path: str = "config.yaml") -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(key_path: str, default: Any = None) -> Any:
    """
    快捷方式：获取配置值
    
    Args:
        key_path: 配置键路径
        default: 默认值
        
    Returns:
        配置值
    """
    return get_config_manager().get(key_path, default)