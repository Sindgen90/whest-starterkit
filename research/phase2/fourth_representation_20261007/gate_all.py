"""Serial invocation of the preserved helper and unchanged official CLI.

Read-only Windows process memory observation every 0.5s; runner is untouched.
"""
import argparse,ctypes,hashlib,json,os,subprocess,sys,time
from ctypes import wintypes as wt
from common import ROOT

def process_memory_tree(root):
    class Entry(ctypes.Structure):
        _fields_=[('dwSize',wt.DWORD),('cntUsage',wt.DWORD),('th32ProcessID',wt.DWORD),('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',wt.DWORD),('cntThreads',wt.DWORD),('th32ParentProcessID',wt.DWORD),('pcPriClassBase',wt.LONG),('dwFlags',wt.DWORD),('szExeFile',wt.WCHAR*260)]
    class Counters(ctypes.Structure):
        _fields_=[('cb',wt.DWORD),('PageFaultCount',wt.DWORD)]+[(x,ctypes.c_size_t) for x in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.windll.kernel32;psapi=ctypes.windll.psapi
    kernel.CreateToolhelp32Snapshot.restype=ctypes.c_void_p;kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    kernel.Process32FirstW.argtypes=[ctypes.c_void_p,ctypes.c_void_p];kernel.Process32NextW.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,wt.DWORD]
    snap=kernel.CreateToolhelp32Snapshot(2,0);e=Entry();e.dwSize=ctypes.sizeof(e);parents={};names={}
    ok=kernel.Process32FirstW(snap,ctypes.byref(e))
    while ok:
        parents[e.th32ProcessID]=e.th32ParentProcessID;names[e.th32ProcessID]=e.szExeFile
        ok=kernel.Process32NextW(snap,ctypes.byref(e))
    kernel.CloseHandle(snap);ids={root}
    while True:
        extended=ids|{pid for pid,parent in parents.items() if parent in ids}
        if extended==ids:break
        ids=extended
    rows=[]
    for pid in ids:
        h=kernel.OpenProcess(0x410,False,pid)
        if not h:continue
        c=Counters();c.cb=ctypes.sizeof(c)
        if psapi.GetProcessMemoryInfo(h,ctypes.byref(c),c.cb):rows.append(dict(pid=pid,name=names.get(pid),peak_working_set_bytes=int(c.PeakWorkingSetSize),working_set_bytes=int(c.WorkingSetSize)))
        kernel.CloseHandle(h)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);a=p.parse_args()
    for name in a.names:
        prefix=ROOT/(name+'_gate');assert not prefix.with_suffix('.json').exists()
        cmd=[sys.executable,str(ROOT.parent/'benchmark_joint_official.py'),'--candidate','fourth_representation_20261007/'+name,'--seeds','20261071','--n-mlps','2','--prefix',str(prefix)]
        log=ROOT/(name+'_gate_stdout.txt');assert not log.exists()
        with log.open('w',encoding='utf-8') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
            observed={};maximum_sum=0;count=0
            while proc.poll() is None:
                rows=process_memory_tree(proc.pid);count+=1;maximum_sum=max(maximum_sum,sum(r['working_set_bytes'] for r in rows))
                for row in rows:
                    pid=row['pid'];observed[pid]=dict(row,peak_working_set_bytes=max(row['peak_working_set_bytes'],observed.get(pid,{}).get('peak_working_set_bytes',0)))
                time.sleep(.5)
            rc=proc.wait()
        memory=dict(observations=count,period_s=.5,max_observed_tree_working_set_bytes=maximum_sum,processes=list(observed.values()),limitation='Sampled process tree, includes CLI/helper/backend/worker. Each process reports OS lifetime peak; can miss short-lived children. No runner or timing-accounting changes.')
        (ROOT/(name+'_gate_memory.json')).write_text(json.dumps(memory,indent=2),encoding='utf-8')
        print(json.dumps(dict(method=name,returncode=rc,max_tree_working_set_bytes=maximum_sum)),flush=True)
if __name__=='__main__':main()
