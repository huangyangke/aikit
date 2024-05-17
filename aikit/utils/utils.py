#!/usr/bin/python3
# -*- coding:utf-8 -*-
import base64
import hashlib
import os
import time
import xxtea
import requests
import oss2
import psutil
import numpy as np
import cv2
from PIL import Image
import requests
import aiohttp
from io import BytesIO
import base64
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

#################################### 文件上传 ####################################
# ref: https://wiki.imgo.tv/pages/viewpage.action?pageId=85775464
class Uploader(object):

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()

    def upload(self, file_path):
        start = time.time()
        resp = self.get_sts_token(file_path)
        if not resp:
            return {}
        sts_token = resp.get('stsToken', {})
        bucket_info = resp.get('bucketInfo', {})
        auth = oss2.StsAuth(sts_token['accessKeyId'], sts_token['accessKeySecret'], sts_token['securityToken'])
        bucket = oss2.Bucket(auth, bucket_info['endpoint'], bucket_info['bucketName'])
        key_info = resp.get('keyInfo', {})
        upload_start = time.time()
        r: oss2.models.PutObjectResult = bucket.put_object_from_file(key_info['key'], file_path)
        upload_cost = time.time() - upload_start
        return {
            'key_info': key_info,
            'oss_put_object_result': vars(r),
            'set_acl': self.set_acl(key_info),
            'url': self.get_url(key_info),
            'upload_cost': upload_cost,
            'total_cost': time.time() - start,
        }

    def get_sts_token(self, file_path):
        req = {
            'platName': self.config['platName'],
            'uploader': self.config['uploader'],
            'fileHeadBase64': self._get_file_head_base64(file_path),
            'fileSize': self._get_file_size(file_path),
            'fileSubfix': self._get_file_ext(file_path),
            'fileContentMd5': self._get_file_md5(file_path),
            'fileCrc64': self._get_file_crc64(file_path),
            'sign': self._get_sign(),
        }
        r = self.session.post(self.config['addr'] + '/cloud/genStsToken', data=req)
        if r.status_code == 200:
            return r.json().get('data', {})
        return {}

    def get_url(self, key_info):
        params = {
            'id': key_info['id'],
            'platName': self.config['platName'],
            'uploader': self.config['uploader'],
            'sign': self._get_sign(),
            'useCdn': self.config['useCdn'],
            'needResize': 0,
        }
        r = self.session.get(self.config['addr'] + '/download/getUrlById', params=params)
        return r.json().get('data', {}).get('url', '')

    def set_acl(self, key_info, acl='public-read'):
        params = {
            'id': key_info['id'],
            'platName': self.config['platName'],
            'uploader': self.config['uploader'],
            'sign': self._get_sign(),
            'acl': acl,
        }
        r = self.session.post(self.config['addr'] + '/acl/setObjectAcl', data=params)
        return r.json()

    def _get_file_head_base64(self, file_path):
        '''
        获取文件前64个字节base64编码
        '''
        with open(file_path, 'rb') as f:
            head_data = f.read(64)
            return base64.b64encode(head_data).decode()

    def _get_file_md5(self, file_path):
        '''
        获取文件md5
        '''
        with open(file_path, 'rb') as f:
            md5_obj = hashlib.md5()
            md5_obj.update(f.read())
            return md5_obj.hexdigest()

    def _get_file_crc64(self, file_path):
        '''
        获取文件内容的crc64
        '''
        with open(file_path, 'rb') as f:
            try:
                import fastcrc
                return fastcrc.crc64.ecma_182(f.read())
            except:
                from crc import Calculator, Crc64
                calculator = Calculator(Crc64.CRC64, optimized=True)
                return calculator.checksum(f.read())

    def _get_file_size(self, file_path):
        '''
        获取文件大小
        '''
        return os.path.getsize(file_path)

    def _get_file_ext(self, file_path):
        '''
        获取文件后缀
        '''
        return os.path.splitext(file_path)[1].replace('.', '')

    def _get_sign(self):
        data = 'mgtv_str_sign(time=%d);' % int(time.time())
        key = '{plat_key}&&{uploader}'.format(plat_key=self.config['platKey'], uploader=self.config['uploader'])
        enc = xxtea.encrypt(data, key)
        return base64.b64encode(enc).decode()

# # 常规oss上传文件
# def upload_file(filepath):
#     bucket_name = os.getenv('BUCKET_NAME')
#     endpoint = os.getenv('ENDPOINT')
#     access_key_id = os.getenv('KEY_ID')
#     access_key_secret = os.getenv('KEY_SECRET')
#     base_dir = os.getenv('ROOT_DIR') or 'platform_oss' # 存储目录
#     source_url = os.getenv('SOURCE_URL')
#     auth = oss2.Auth(access_key_id, access_key_secret)
#     bucket = oss2.Bucket(auth, endpoint, bucket_name)
#     _, ext = os.path.splitext(filepath) # 后缀
#     target_name = datetime.now().strftime('%Y%m%d%H%M%S%f') + ext
#     target_file = os.path.join(base_dir, 'dm-art', target_name)
#     bucket.put_object_from_file(target_file, filepath)
#     return source_url + target_file

#################################### 时间计算器 ####################################
# 时间计算器
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1.0 #1000.0
        self.time_unit = 's' #"ms"

    def start(self, name: str) -> None:
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")
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
@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def downloadfile(url, save_path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(save_path, 'wb') as file:
        for data in resp.iter_content(chunk_size=1024):
            file.write(data)
    resp.close()
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
    uploader = Uploader({
        'addr': 'http://10.200.19.32:10010',
        'platName': "d11m_ai_work",
        'platKey': "iT4zP1mM3zH9nL6i",
        'uploader': "hyk",
        'useCdn': 1,
    })

    file = '/mnt/cluster/03_dataset/images/1.jpg'
    ret = uploader.upload(file)
    print(ret['url'], ret['key_info']['id'])