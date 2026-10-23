"""Pinned, node-local bootstrap for the optional native temporal extension.

Only fixed trusted-release bytes are loaded, after embedded SHA256 checks.
No pip, site-packages writes, main Kitchen replacement, or sampling-time build.
"""
import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
import urllib.request

SOURCE_COMMIT = '9953b6caec7a5cb3e2486dd97a365f7af44eedad'
RAW = f'https://raw.githubusercontent.com/alisson-anjos/comfy-kitchen/{SOURCE_COMMIT}'
RELEASE = 'https://github.com/alisson-anjos/comfy-kitchen/releases/download/h3-native-temporal-kernel-v1'
BINARY_HASH = 'ad87534bfb3ece2af84c201e326b2b405f0108f34efb36937c09967c14f0329d'
ASSETS = (
    ('temporal_attention.py', RAW+'/comfy_kitchen/temporal_attention.py', '0151e8b5b5886c9e8b4608cbc179e9dd6f934170cf45b8d27e14801f3c4828a1', 256*1024),
    ('backends/cuda/native_temporal/libck_temporal_sm120.so', RELEASE+'/libck_temporal_sm120.so', BINARY_HASH, 16*1024*1024),
    ('LICENSE', RAW+'/LICENSE', 'cd1c9ea1573c9047e9b5a3676ecc1c8f755e56f249ac1ebd8b5ceda7899037da', 64*1024),
    ('NOTICE', RAW+'/NOTICE', '35c43127ad3acbd6bc94ff63c56684a7c26eedf07795e596c8105a0476dae2ea', 64*1024),
)
_LOCK = threading.Lock()
_READY = {}
_FAILED = set()


def cache_root():
    explicit=os.environ.get('BFS_H3_TEMPORAL_CACHE_DIR')
    if explicit:
        base=Path(explicit).expanduser()
    else:
        try:
            import folder_paths
            base=Path(folder_paths.get_user_directory())/'bfs'/'h3_temporal_kernels'
        except (ImportError,AttributeError):
            base=Path.home()/'.cache'/'bfsnodes'/'h3_temporal_kernels'
    return base/('v1-linux-x86_64-sm120-'+BINARY_HASH[:12])


def compatible(device):
    if os.environ.get('BFS_H3_TEMPORAL_AUTO_INSTALL','1').lower() in ('0','false','off'):
        return False,'automatic setup disabled'
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
        return False,'automatic prebuilt supports Linux x86_64 only'
    import torch
    if getattr(torch.version,'hip',None) or not torch.cuda.is_available() or torch.cuda.get_device_capability(device)!=(12,0):
        return False,'automatic prebuilt supports NVIDIA SM120 only'
    # Check the actual driver, not torch.version.cuda: the sidecar uses its own
    # static CUDA runtime and the same device/stream primary context.
    driver=ctypes.CDLL('libcuda.so.1');version=ctypes.c_int()
    if driver.cuDriverGetVersion(ctypes.byref(version))!=0 or version.value<13000:
        return False,'the prebuilt requires a driver supporting CUDA 13'
    import comfy_kitchen
    if not comfy_kitchen.int8_attention_is_available(device):
        return False,'Kitchen native INT8 attention unavailable'
    from comfy_kitchen.sage_attention import prequantize_int8_attention
    if not callable(prequantize_int8_attention):
        return False,'Kitchen prequantization API unavailable'
    return True,None


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def download(path,url,expected,limit):
    if path.is_file() and digest(path)==expected:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    temp=Path(name)
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'BFSNodes-H3-Temporal/1'})
        h=hashlib.sha256();total=0
        with os.fdopen(fd,'wb') as out,urllib.request.urlopen(request,timeout=20) as response:
            while True:
                block=response.read(256*1024)
                if not block:break
                total+=len(block)
                if total>limit:raise RuntimeError('Temporal asset exceeds the pinned size limit')
                h.update(block);out.write(block)
        if h.hexdigest()!=expected:
            raise RuntimeError('Temporal asset checksum mismatch; downloaded code was not loaded')
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def provision(root):
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.install.lock').open('a+b') as lock:
        deadline=time.monotonic()+60
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for temporal setup lock')
                time.sleep(.1)
        missing=any(not (root/rel).is_file() or digest(root/rel)!=sha for rel,_,sha,_ in ASSETS)
        if missing:
            print('[BFS H3 native temporal] First-use setup: downloading the verified one-pass extension (~3.3 MB). No compilation or pip install.',flush=True)
        for rel,url,sha,limit in ASSETS:
            download(root/rel,url,sha,limit)
        # Verify the native binary before executing the Python entry point too.
        binary=root/ASSETS[1][0];lib=ctypes.CDLL(str(binary))
        lib.ck_temporal_abi_version.restype=ctypes.c_int;lib.ck_temporal_arch.restype=ctypes.c_int
        if lib.ck_temporal_abi_version()!=1 or lib.ck_temporal_arch()!=120:
            raise RuntimeError('Temporal prebuilt ABI/architecture mismatch')
        name='_bfs_ck_temporal_v1_sm120_'+ASSETS[0][2][:12]+'_'+hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
        if name in sys.modules:
            return sys.modules[name]
        spec=importlib.util.spec_from_file_location(name,root/'temporal_attention.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name,None);raise
        if getattr(module,'VERSION',None)!='h3_native_temporal_kernel_v1':
            sys.modules.pop(name,None);raise RuntimeError('Temporal API version mismatch')
        print(f'[BFS H3 native temporal] Verified one-pass extension ready: {root}',flush=True)
        return module


def ensure(device):
    """At most one automatic setup attempt per process/cache location."""
    key=(str(cache_root()),str(device))
    with _LOCK:
        if key in _READY:return _READY[key]
        if key in _FAILED:return None
        try:
            ok,reason=compatible(device)
            if not ok:
                _FAILED.add(key)
                print(f'[BFS H3 native temporal] Automatic native setup skipped: {reason}; retaining the existing backend.',flush=True)
                return None
            module=provision(Path(key[0]))
            if not module.is_available(device):raise RuntimeError('Verified native extension is unavailable for this device')
            _READY[key]=module
            return module
        except Exception as exc:
            _FAILED.add(key)
            print(f'[BFS H3 native temporal] Automatic native setup unavailable: {exc}. Retaining the existing backend. Restart ComfyUI to retry, or install the extension manually.',flush=True)
            return None
