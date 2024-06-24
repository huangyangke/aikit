#!/usr/bin/python3
# -*- coding:utf-8 -*-
import base64
import sys
import logging
from types import FrameType
from typing import cast
from loguru import logger
import os
import time
import requests
import urllib3
import psutil
import numpy as np
import cv2
from PIL import Image
import aiohttp
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_fixed
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

#################################### 日志 ####################################
# 清空所有设置
logger.remove()
# 添加控制台输出的格式,sys.stdout为输出到屏幕;关于这些配置还需要自定义请移步官网查看相关参数说明
logger.add(sys.stdout,
    format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "  # 颜色>时间
            "{process.name} | "  # 进程名
            "{thread.name} | "  # 进程名
            "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
            ":<cyan>{line}</cyan> | "  # 行号
            "<level>{level}</level>: "  # 等级
            "<level>{message}</level>",  # 日志内容
)

def add_logfile(logdir):
    # 文件的命名
    log_path = os.path.join(logdir, "log_{time:YYYY-MM-DD}.log")
    # 判断日志文件夹是否存在，不存则创建
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    # 日志写入文件
    logger.add(log_path,  # 写入目录指定文件
        format='{time:YYYYMMDD HH:mm:ss} - '  # 时间
                "{process.name} | "  # 进程名
                "{thread.name} | "  # 进程名
                '{module}.{function}:{line} - {level} -{message}',  # 模块名.方法名:行号
        encoding='utf-8',
        backtrace=True,  # 回溯
        diagnose=True,  # 诊断
        enqueue=True,  # 异步写入
        rotation="00:00",  # 每日更新时间
        # retention='7 days',  # 设置历史保留时长
        # rotation="5kb",  # 切割，设置文件大小，rotation="12:00"，rotation="1 week"
        # filter="my_module"  # 过滤模块
        # compression="zip"   # 文件压缩
    )
    return logger

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
 
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1
 
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
                                  
def replace_fastapi_log():
    """
    使用方案：将uvicorn输出的全部让loguru管理
    config = uvicorn.Config(app, host='0.0.0.0', port=port, workers=1)
    replace_fastapi_log()
    uvicorn.Server(config).run()
    """
    LOGGER_NAMES = ("uvicorn.asgi", "uvicorn.access", "uvicorn")
    # change handler for default uvicorn logger
    for logger_name in LOGGER_NAMES:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
#################################### 时间计算器 ####################################
# 时间计算器
class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1.0 #1000.0
        self.time_unit = 's' #"ms"

    def start(self, name: str) -> None:
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        self.items[name] = time.time()
        logger.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logger.info(f"{name} finished in {t:.2f}{self.time_unit}.")
        
timer = Timer() 

def timer_decorator(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer.start(name)
            result = func(*args, **kwargs)
            timer.end(name)
            return result
        return wrapper
    return decorator
#################################### 文件下载 ####################################
# requests性能更好
@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def downloadfile(url, save_path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(save_path, 'wb') as file:
        for data in resp.iter_content(chunk_size=1024):
            file.write(data)
    resp.close()
    
# @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))    
# def downloadfile(url, save_path):
#     http = urllib3.PoolManager()
#     response = http.request('GET', url, preload_content=False)
#     # 以二进制写模式打开文件
#     with open(save_path, 'wb') as file:
#         while True:
#             data = response.read(1024)
#             if not data: break
#             file.write(data)
#     response.close()   
#################################### 图片下载 ####################################
# 协程函数 提高性能
async def get_image_data_from_http(image_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status != 200:
                return None
            image_data = BytesIO(await response.read())    
    return image_data

# 图片加载，支持同步和异步操作
# 其中异步操作可以用于fastapi等支持协程的框架中
@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def get_image_from_url(image_data,form='opencv'):
    try:
        if image_data is None: return False, None
        if 'http' in image_data:
            response = requests.get(image_data)
            if response.status_code != 200: return False, None        
            image_data = BytesIO(response.content)        
        if form == 'base64':
            image = base64.b64decode(image_data.getvalue()).decode('utf-8')
        else:
            # 将获取的数据转换为二进制流
            if form == 'opencv':
                image = cv2.imdecode(np.frombuffer(image_data.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                image = Image.open(image_data) 
        return True, image
    except:
        return False, None
#################################### 图片编解码 ####################################
# 图片base64编解码
def B64ImageEncode(image_array,types='.jpg'):
    rect,image_buffer=cv2.imencode(types, image_array)
    image_b64=base64.b64encode(image_buffer)
    image_b64=image_b64.decode('utf-8')
    return image_b64

def BufferImageDecode(image_buffer):
    nparr = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return image

def B64ImageDecode(image_b64):
    image_buffer = base64.b64decode(image_b64)
    return BufferImageDecode(image_buffer)    
    
#################################### 获取当前进程内存占用 ####################################
def get_current_memory_gb():
    # 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return round(info.uss / 1024. / 1024. /1024., 2)
      
if __name__ == '__main__':
    # timer.start('下载图片')
    # for i in range(1):
    #     downloadfile(url='https://img1.baidu.com/it/u=1901146814,3537581211&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=734', 
    #                 save_path='test.jpg')
    # timer.end('下载图片')
    # replace_fastapi_log()
    logger = add_logfile('log')
    logger.info('你好')