"""Signal only the birth-verified owned E2 worker through a root-owned pidfd."""
import os,sys,signal,ctypes
def syscall(number,*args):
 if os.uname().machine!="x86_64":raise RuntimeError("Unsupported pidfd ABI")
 libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
 result=libc.syscall(ctypes.c_long(number),*args)
 if result<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
 return result

def main():
 pid,birth=map(int,sys.argv[1:3])
 try:
  fd=syscall(434,ctypes.c_int(pid),ctypes.c_uint(0))
 except ProcessLookupError:return
 try:
  actual=int(open('/proc/%d/stat'%pid).read().split(') ',1)[1].split()[19])
  if actual!=birth:raise RuntimeError('Worker birth mismatch; signal refused')
  syscall(424,ctypes.c_int(fd),ctypes.c_int(signal.SIGKILL),ctypes.c_void_p(),ctypes.c_uint(0))
 finally:os.close(fd)
if __name__=='__main__':main()
