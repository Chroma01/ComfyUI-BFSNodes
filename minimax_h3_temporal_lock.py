"""Per-cache process lock on Linux and Windows."""
from contextlib import contextmanager
import platform
import time

@contextmanager
def file_lock(path,timeout=60):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        windows=platform.system()=='Windows'
        if windows:
            import msvcrt
            handle.seek(0,2)
            if handle.tell()==0:handle.write(b'\0');handle.flush()
            def acquire():handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            def release():handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            def acquire():fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def release():fcntl.flock(handle,fcntl.LOCK_UN)
        deadline=time.monotonic()+timeout
        while True:
            try:acquire();break
            except OSError:
                if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for temporal kernel setup')
                time.sleep(.1)
        try:yield
        finally:release()
