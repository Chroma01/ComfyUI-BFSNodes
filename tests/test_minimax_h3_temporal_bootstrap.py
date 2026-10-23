import concurrent.futures,hashlib,io,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minimax_h3_temporal_bootstrap as boot


class NativeFunction:
 def __init__(self,result):self.result=result
 def __call__(self):return self.result

@pytest.fixture
def setup(monkeypatch,tmp_path):
 monkeypatch.setenv('BFS_H3_TEMPORAL_CACHE_DIR',str(tmp_path))
 boot._READY.clear();boot._FAILED.clear()
 source=b"VERSION='h3_native_temporal_kernel_v1'\ndef is_available(device):return True\n"
 binary=b'fake verified native fixture'
 target=dict(platform='Linux',arch=120,file='libck_temporal_sm120.so',native_abi=1,sha256=hashlib.sha256(binary).hexdigest(),bytes=len(binary))
 monkeypatch.setattr(boot,'COMMON_ASSETS',(('temporal_attention.py','https://test.invalid/api',hashlib.sha256(source).hexdigest(),1024),))
 monkeypatch.setattr(boot,'binary_for_device',lambda d:target)
 requests=[]
 def urlopen(request,timeout):
  requests.append(request.full_url)
  return io.BytesIO(source if request.full_url.endswith('/api') else binary)
 monkeypatch.setattr(boot.urllib.request,'urlopen',urlopen)
 loads=[]
 def cdll(path):
  loads.append(path)
  return SimpleNamespace(ck_temporal_abi_version=NativeFunction(1),ck_temporal_arch=NativeFunction(target['arch']))
 monkeypatch.setattr(boot.ctypes,'CDLL',cdll)
 monkeypatch.setattr(boot,'compatible',lambda d:(True,None))
 yield SimpleNamespace(root=boot.cache_root(target),target=target,requests=requests,loads=loads,source=source,binary=binary)
 boot._READY.clear();boot._FAILED.clear()


def test_first_use_installs_once_and_preserves_current_environment(setup):
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  results=list(pool.map(boot.ensure,['cuda:0']*4))
 assert all(m is results[0] for m in results)
 assert len(setup.requests)==2 and len(setup.loads)==1
 assert (setup.root/'temporal_attention.py').read_bytes()==setup.source
 assert not list(setup.root.rglob('*.partial'))
 assert str(results[0].__file__).startswith(str(setup.root))


def test_cached_bytes_work_without_network_after_restart(setup,monkeypatch):
 assert boot.ensure('cuda:0') is not None
 boot._READY.clear();boot._FAILED.clear()
 def offline(*a,**kw):raise AssertionError('valid cache must not access the network')
 monkeypatch.setattr(boot.urllib.request,'urlopen',offline)
 assert boot.ensure('cuda:0') is not None


def test_corrupted_download_never_loads_or_executes_code(setup,monkeypatch):
 monkeypatch.setattr(boot.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(b'wrong bytes'))
 assert boot.ensure('cuda:0') is None
 assert setup.loads==[]
 assert not (setup.root/'temporal_attention.py').exists()
 assert not list(setup.root.rglob('*.partial'))


def test_failure_is_not_retried_every_attention_block(setup,monkeypatch):
 calls=[]
 def broken(*a,**kw):calls.append(1);raise OSError('offline')
 monkeypatch.setattr(boot.urllib.request,'urlopen',broken)
 assert boot.ensure('cuda:0') is None
 assert boot.ensure('cuda:0') is None
 assert len(calls)==1


def test_bad_existing_cache_is_replaced_only_after_verified_download(setup):
 setup.root.mkdir(parents=True)
 (setup.root/'temporal_attention.py').write_text('corrupt')
 assert boot.ensure('cuda:0') is not None
 assert (setup.root/'temporal_attention.py').read_bytes()==setup.source


def test_size_limit_aborts_and_cleans_partial_file(tmp_path,monkeypatch):
 monkeypatch.setattr(boot.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(b'12345'))
 with pytest.raises(RuntimeError,match='size limit'):boot.download(tmp_path/'asset','https://test.invalid/asset','ignored',2)
 assert list(tmp_path.iterdir())==[]


def test_unsupported_device_keeps_existing_backend_without_download(setup,monkeypatch):
 monkeypatch.setattr(boot,'compatible',lambda d:(False,'unsupported GPU'))
 assert boot.ensure('cuda:0') is None
 assert setup.requests==[]


def test_device_failure_does_not_block_supported_device(setup,monkeypatch):
 monkeypatch.setattr(boot,'compatible',lambda d:(d=='cuda:1','unsupported GPU'))
 assert boot.ensure('cuda:0') is None
 assert boot.ensure('cuda:1') is not None


def test_opt_out_is_checked_before_gpu_or_network(monkeypatch):
 monkeypatch.setenv('BFS_H3_TEMPORAL_AUTO_INSTALL','0')
 assert boot.compatible('cuda:0')==(False,'automatic setup disabled')


def test_bad_binary_arch_is_rejected_before_python_api_execution(setup,monkeypatch):
 before=set(sys.modules)
 monkeypatch.setattr(boot.ctypes,'CDLL',lambda p:SimpleNamespace(ck_temporal_abi_version=NativeFunction(1),ck_temporal_arch=NativeFunction(89)))
 assert boot.ensure('cuda:0') is None
 assert not [n for n in set(sys.modules)-before if n.startswith('_bfs_ck_temporal')]


@pytest.mark.parametrize('system,arch',[(b['platform'],b['arch']) for b in boot.MANIFEST['binaries']])
def test_release_matrix_selects_exact_os_and_arch(system,arch):
 target=boot.select_binary(system,'AMD64' if system=='Windows' else 'x86_64',(arch//10,arch%10))
 assert target['arch']==arch and target['platform']==system
 assert target['file'].endswith('.dll' if system=='Windows' else '.so')
 assert len(target['sha256'])==64 and target['bytes']>0 and target['native_abi']==1
 assert target['gpu_runtime_validation'] is False


@pytest.mark.parametrize('system,machine,cap',[('Darwin','x86_64',(12,0)),('Linux','aarch64',(12,0)),('Windows','AMD64',(9,0)),('Linux','x86_64',(7,5))])
def test_unpublished_combinations_never_substitute_another_binary(system,machine,cap):
 assert boot.select_binary(system,machine,cap) is None


def test_windows_ada_downloads_only_its_own_dll(setup,monkeypatch):
 from contextlib import nullcontext
 setup.target.update(platform='Windows',arch=89,file='libck_temporal_sm89.dll')
 monkeypatch.setattr(boot.platform,'system',lambda:'Windows')
 monkeypatch.setattr(boot,'file_lock',lambda *a,**k:nullcontext())
 module=boot.ensure('cuda:0')
 assert module is not None
 assert len(setup.requests)==2
 assert any(url.endswith('libck_temporal_sm89.dll') for url in setup.requests)
 assert not any('sm120' in url or url.endswith('.so') for url in setup.requests)
 assert (boot.cache_root(setup.target)/'backends/cuda/native_temporal/libck_temporal_sm89.dll').is_file()


def test_platform_and_arch_produce_separate_cache_names(setup):
 linux=boot.cache_root(setup.target)
 windows=boot.cache_root(dict(setup.target,platform='Windows'))
 ada=boot.cache_root(dict(setup.target,arch=89))
 assert len({linux,windows,ada})==3


def test_automatic_setup_never_launches_a_compiler_or_pip(setup,monkeypatch):
 import subprocess
 def forbidden(*a,**kw):raise AssertionError('automatic download must not launch a build or pip')
 monkeypatch.setattr(subprocess,'Popen',forbidden)
 assert boot.ensure('cuda:0') is not None


def test_windows_driver_query_uses_nvcuda_not_linux_lib(monkeypatch):
 seen=[]
 class Driver:
  def cuDriverGetVersion(self,p):p._obj.value=13000;return 0
 monkeypatch.setattr(boot.platform,'system',lambda:'Windows')
 monkeypatch.setattr(boot.ctypes,'WinDLL',lambda name:(seen.append(name) or Driver()),raising=False)
 assert boot.driver_version()==13000 and seen==['nvcuda.dll']
