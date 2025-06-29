#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志系统设置模块

功能：
1. 统一的日志配置
2. 文件和控制台日志处理
3. 日志轮转和格式化
4. 性能日志记录
"""

import os
import logging
import logging.handlers
from typing import Optional
from config_manager import get_config_manager


class LoggerSetup:
    """
    日志设置类
    """
    
    def __init__(self, config_manager=None):
        """
        初始化日志设置
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager or get_config_manager()
        self.loggers = {}
    
    def setup_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        设置并返回指定名称的日志器
        
        Args:
            name: 日志器名称
            level: 日志级别
            
        Returns:
            配置好的日志器
        """
        if name in self.loggers:
            return self.loggers[name]
        
        # 获取日志配置
        log_config = self.config_manager.get_logging_config()
        
        # 创建日志器
        logger = logging.getLogger(name)
        
        # 设置日志级别
        log_level = level or log_config.get('level', 'INFO')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(
            log_config.get('format', 
                          '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # 设置控制台处理器
        console_config = log_config.get('console_handler', {})
        if console_config.get('enabled', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 设置文件处理器
        file_config = log_config.get('file_handler', {})
        if file_config.get('enabled', True):
            # 确保日志目录存在
            log_file = file_config.get('filename', 'logs/audio_classifier.log')
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # 创建轮转文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=file_config.get('max_bytes', 10485760),  # 10MB
                backupCount=file_config.get('backup_count', 5),
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 防止日志重复
        logger.propagate = False
        
        # 缓存日志器
        self.loggers[name] = logger
        
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            日志器实例
        """
        if name not in self.loggers:
            return self.setup_logger(name)
        return self.loggers[name]


# 全局日志设置实例
_logger_setup = None


def get_logger_setup() -> LoggerSetup:
    """
    获取全局日志设置实例
    
    Returns:
        日志设置实例
    """
    global _logger_setup
    if _logger_setup is None:
        _logger_setup = LoggerSetup()
    return _logger_setup


def get_logger(name: str) -> logging.Logger:
    """
    快捷方式：获取日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    return get_logger_setup().get_logger(name)


def setup_root_logger() -> None:
    """
    设置根日志器
    """
    logger_setup = get_logger_setup()
    logger_setup.setup_logger('root')


class PerformanceLogger:
    """
    性能日志记录器
    """
    
    def __init__(self, logger_name: str = 'performance'):
        """
        初始化性能日志记录器
        
        Args:
            logger_name: 日志器名称
        """
        self.logger = get_logger(logger_name)
        self.config_manager = get_config_manager()
    
    def log_performance(self, operation: str, duration: float, 
                       additional_info: Optional[dict] = None) -> None:
        """
        记录性能信息
        
        Args:
            operation: 操作名称
            duration: 持续时间（秒）
            additional_info: 额外信息
        """
        perf_config = self.config_manager.get_performance_config()
        
        if not perf_config.get('monitoring_enabled', True):
            return
        
        # 构建日志消息
        message = f"性能监控 - {operation}: {duration:.4f}秒"
        
        if additional_info:
            info_str = ", ".join([f"{k}={v}" for k, v in additional_info.items()])
            message += f" ({info_str})"
        
        self.logger.info(message)
        
        # 保存到性能指标文件
        if perf_config.get('save_metrics', False):
            self._save_metric_to_file(operation, duration, additional_info)
    
    def _save_metric_to_file(self, operation: str, duration: float, 
                            additional_info: Optional[dict] = None) -> None:
        """
        保存性能指标到文件
        
        Args:
            operation: 操作名称
            duration: 持续时间
            additional_info: 额外信息
        """
        import json
        import time
        
        perf_config = self.config_manager.get_performance_config()
        metrics_file = perf_config.get('metrics_file', 'logs/performance_metrics.json')
        
        # 确保目录存在
        metrics_dir = os.path.dirname(metrics_file)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)
        
        # 构建指标数据
        metric_data = {
            'timestamp': time.time(),
            'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
            'operation': operation,
            'duration': duration,
            'additional_info': additional_info or {}
        }
        
        try:
            # 读取现有数据
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            else:
                metrics = []
            
            # 添加新指标
            metrics.append(metric_data)
            
            # 保持最近1000条记录
            if len(metrics) > 1000:
                metrics = metrics[-1000:]
            
            # 写入文件
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"保存性能指标失败: {e}")


# 全局性能日志记录器
_performance_logger = None


def get_performance_logger() -> PerformanceLogger:
    """
    获取全局性能日志记录器
    
    Returns:
        性能日志记录器实例
    """
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


def log_performance(operation: str, duration: float, 
                   additional_info: Optional[dict] = None) -> None:
    """
    快捷方式：记录性能信息
    
    Args:
        operation: 操作名称
        duration: 持续时间（秒）
        additional_info: 额外信息
    """
    get_performance_logger().log_performance(operation, duration, additional_info)