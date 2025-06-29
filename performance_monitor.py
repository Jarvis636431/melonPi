#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控模块

功能：
1. 函数执行时间监控装饰器
2. 上下文管理器性能监控
3. 内存使用监控
4. 系统资源监控
"""

import time
import functools
import psutil
import threading
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager
from logger_setup import get_logger, log_performance


class PerformanceMonitor:
    """
    性能监控器类
    """
    
    def __init__(self, logger_name: str = 'performance_monitor'):
        """
        初始化性能监控器
        
        Args:
            logger_name: 日志器名称
        """
        self.logger = get_logger(logger_name)
        self._monitoring_active = False
        self._monitor_thread = None
        self._system_metrics = []
    
    def timing_decorator(self, operation_name: Optional[str] = None, 
                        log_args: bool = False, log_result: bool = False):
        """
        函数执行时间监控装饰器
        
        Args:
            operation_name: 操作名称，默认使用函数名
            log_args: 是否记录函数参数
            log_result: 是否记录函数返回值
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # 确定操作名称
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # 记录开始时间和内存
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                # 构建额外信息
                additional_info = {
                    'start_memory_mb': start_memory
                }
                
                if log_args and args:
                    additional_info['args_count'] = len(args)
                if log_args and kwargs:
                    additional_info['kwargs_count'] = len(kwargs)
                
                try:
                    # 执行函数
                    result = func(*args, **kwargs)
                    
                    # 记录结束时间和内存
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    duration = end_time - start_time
                    
                    # 更新额外信息
                    additional_info.update({
                        'end_memory_mb': end_memory,
                        'memory_delta_mb': end_memory - start_memory,
                        'status': 'success'
                    })
                    
                    if log_result and result is not None:
                        additional_info['result_type'] = type(result).__name__
                        if hasattr(result, '__len__'):
                            additional_info['result_length'] = len(result)
                    
                    # 记录性能信息
                    log_performance(op_name, duration, additional_info)
                    
                    return result
                    
                except Exception as e:
                    # 记录异常情况
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    additional_info.update({
                        'status': 'error',
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    })
                    
                    log_performance(f"{op_name}_error", duration, additional_info)
                    raise
            
            return wrapper
        return decorator
    
    @contextmanager
    def monitor_operation(self, operation_name: str, 
                         additional_info: Optional[Dict] = None):
        """
        上下文管理器形式的性能监控
        
        Args:
            operation_name: 操作名称
            additional_info: 额外信息
            
        Yields:
            监控上下文
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        info = additional_info or {}
        info['start_memory_mb'] = start_memory
        
        try:
            yield
            
            # 成功完成
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration = end_time - start_time
            
            info.update({
                'end_memory_mb': end_memory,
                'memory_delta_mb': end_memory - start_memory,
                'status': 'success'
            })
            
            log_performance(operation_name, duration, info)
            
        except Exception as e:
            # 异常情况
            end_time = time.time()
            duration = end_time - start_time
            
            info.update({
                'status': 'error',
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            
            log_performance(f"{operation_name}_error", duration, info)
            raise
    
    def _get_memory_usage(self) -> float:
        """
        获取当前进程内存使用量（MB）
        
        Returns:
            内存使用量（MB）
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # 转换为MB
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """
        获取当前进程CPU使用率
        
        Returns:
            CPU使用率（百分比）
        """
        try:
            process = psutil.Process()
            return process.cpu_percent()
        except Exception:
            return 0.0
    
    def start_system_monitoring(self, interval: float = 5.0) -> None:
        """
        开始系统资源监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring_active:
            self.logger.warning("系统监控已在运行")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"开始系统资源监控，间隔: {interval}秒")
    
    def stop_system_monitoring(self) -> None:
        """
        停止系统资源监控
        """
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        self.logger.info("系统资源监控已停止")
    
    def _system_monitor_loop(self, interval: float) -> None:
        """
        系统监控循环
        
        Args:
            interval: 监控间隔
        """
        while self._monitoring_active:
            try:
                # 获取系统指标
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 获取进程指标
                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024  # MB
                process_cpu = process.cpu_percent()
                
                # 记录系统指标
                system_info = {
                    'system_cpu_percent': cpu_percent,
                    'system_memory_percent': memory.percent,
                    'system_memory_available_gb': memory.available / 1024 / 1024 / 1024,
                    'system_disk_percent': disk.percent,
                    'process_memory_mb': process_memory,
                    'process_cpu_percent': process_cpu
                }
                
                log_performance("system_monitoring", interval, system_info)
                
                # 保存到内存中的指标列表
                self._system_metrics.append({
                    'timestamp': time.time(),
                    **system_info
                })
                
                # 保持最近100条记录
                if len(self._system_metrics) > 100:
                    self._system_metrics = self._system_metrics[-100:]
                
            except Exception as e:
                self.logger.error(f"系统监控出错: {e}")
            
            time.sleep(interval)
    
    def get_recent_metrics(self, count: int = 10) -> list:
        """
        获取最近的系统指标
        
        Args:
            count: 返回的记录数量
            
        Returns:
            最近的系统指标列表
        """
        return self._system_metrics[-count:] if self._system_metrics else []
    
    def log_memory_usage(self, operation: str) -> None:
        """
        记录当前内存使用情况
        
        Args:
            operation: 操作描述
        """
        memory_mb = self._get_memory_usage()
        cpu_percent = self._get_cpu_usage()
        
        additional_info = {
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent
        }
        
        self.logger.info(f"内存使用情况 - {operation}: {memory_mb:.2f}MB, CPU: {cpu_percent:.1f}%")
        log_performance(f"memory_check_{operation}", 0.0, additional_info)


# 全局性能监控器实例
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """
    获取全局性能监控器实例
    
    Returns:
        性能监控器实例
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


# 便捷装饰器函数
def monitor_performance(operation_name: Optional[str] = None, 
                       log_args: bool = False, log_result: bool = False):
    """
    性能监控装饰器（便捷函数）
    
    Args:
        operation_name: 操作名称
        log_args: 是否记录参数
        log_result: 是否记录结果
        
    Returns:
        装饰器函数
    """
    return get_performance_monitor().timing_decorator(
        operation_name, log_args, log_result
    )


def monitor_operation(operation_name: str, additional_info: Optional[Dict] = None):
    """
    操作监控上下文管理器（便捷函数）
    
    Args:
        operation_name: 操作名称
        additional_info: 额外信息
        
    Returns:
        上下文管理器
    """
    return get_performance_monitor().monitor_operation(operation_name, additional_info)


def log_memory_usage(operation: str) -> None:
    """
    记录内存使用情况（便捷函数）
    
    Args:
        operation: 操作描述
    """
    get_performance_monitor().log_memory_usage(operation)