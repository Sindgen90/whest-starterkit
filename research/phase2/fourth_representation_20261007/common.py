import ctypes,importlib.util,time
from pathlib import Path
import numpy as np
import flopscope as flops
from whestbench.scoring import materialise_predictions
ROOT=Path(__file__).resolve().parent
def load(name):
    p=ROOT/name/'estimator.py';spec=importlib.util.spec_from_file_location('fourth_'+name,p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def peak_memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    p=Counters();p.cb=ctypes.sizeof(p)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb):raise ctypes.WinError()
    return int(p.PeakWorkingSetSize)
def measure(est,mlp,diagnostics=False):
    est.collect_diagnostics=diagnostics;start=time.perf_counter()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
        pred=est.predict(mlp,2**41)
        arr=np.asarray(materialise_predictions(pred,depth=mlp.depth,width=mlp.width),dtype=np.float64)
    states=[{k:(np.asarray(v) if v is not None else None) for k,v in s.items()} for s in getattr(est,'states',[])]
    timing=budget.summary_dict()
    return arr,dict(flops=int(budget.flops_used),wall_s=time.perf_counter()-start,residual_s=timing['residual_wall_time_s'],process_peak_working_set_bytes=peak_memory()),states
def crossrisk(errors):
    e=np.asarray(errors);s=len(e)
    return float(np.mean((np.sum(e,axis=0)**2-np.sum(e*e,axis=0))/(s*(s-1))))
def interval(values):
    v=np.asarray(values);mean=float(np.mean(v));se=float(np.std(v,ddof=1)/np.sqrt(len(v)))
    return dict(mean=mean,lower95=mean-3.182446*se,upper95=mean+3.182446*se)
def risk_interval(errors):
    e=np.asarray(errors);n=len(e);full=crossrisk(e)
    pseudo=[n*full-(n-1)*crossrisk(np.delete(e,i,axis=0)) for i in range(n)]
    out=interval(pseudo);out['estimate']=full;out['method']='delete-one-stream U-statistic jackknife, t3; approximate with only four streams'
    return out
