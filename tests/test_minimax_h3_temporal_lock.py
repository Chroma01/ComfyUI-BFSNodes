import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minimax_h3_temporal_lock as locks


def test_windows_locks_one_existing_byte(tmp_path,monkeypatch):
 calls=[]
 monkeypatch.setattr(locks.platform,'system',lambda:'Windows')
 monkeypatch.setitem(sys.modules,'msvcrt',SimpleNamespace(LK_NBLCK=1,LK_UNLCK=2,locking=lambda fd,mode,size:calls.append((mode,size))))
 path=tmp_path/'cache.lock'
 with locks.file_lock(path):assert path.stat().st_size>=1
 assert calls==[(1,1),(2,1)]


def test_linux_lock_acquires_and_releases(tmp_path):
 with locks.file_lock(tmp_path/'cache.lock'):pass
 with locks.file_lock(tmp_path/'cache.lock'):pass
