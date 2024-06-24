# AI Kit
AI 公共工具包

# INSTALL
+ 为确保安装成功，建议先升级setuptools: `pip install --upgrade setuptools`
+ 有Git权限安装方式：`pip install -U git+https://github.com/huangyangke/aikit.git`
+ 无Git权限安装方式：`pip install -U git+https://huangyangke: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@github.com/huangyangke/aikit.git`

ref:https://docs.readthedocs.io/en/stable/guides/private-python-packages.html

# Usage
## uploader
```python
from aikit.utils import uploader

config = {
    'addr': 'http://10.200.19.32:10010', # 上传服务地址
    'platName': 'xx', # 上传服务用户名
    'platKey': 'zz',  # 上传服务密钥
    'uploader': 1111, # 上传者id
    'useCdn': 0,      # 是否使用CDN
}
the_uploader = uploader.Uploader(config)
ret = the_uploader.upload(file)
# url and file id
print(ret['url'], ret['key_info']['id'])
```