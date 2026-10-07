"""Serial final checks, with append-only logs and no runner alterations."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(script,args,logname):
    path=ROOT/logname;assert not path.exists()
    print(json.dumps(dict(starting=script,args=args)),flush=True);started=time.perf_counter()
    with path.open('w',encoding='utf-8') as f:
        result=subprocess.run([sys.executable,'-W','ignore',str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    print(json.dumps(dict(completed=script,returncode=result.returncode,wall_s=time.perf_counter()-started)),flush=True)
    assert result.returncode==0,logname
def main():
    assert all((ROOT/(name+'_gate.json')).exists() for name in ('baseline_snapshot','selective_block8','qr','block8','block16','qblock8','qblock16','qrblock8'))
    run('full_width.py',['--names','selective_block8_compat','--output','compat_full_width_results.json'],'compat_full_width_stdout.txt')
    run('gate_all.py',['--names','selective_block8_compat'],'compat_gate_controller_stdout.txt')
    gate=json.loads((ROOT/'selective_block8_compat_gate.json').read_text())
    if all(r['summary']['n_failed_mlps']==0 for r in gate['records']):run('package_candidate.py',['selective_block8_compat'],'compat_packaging_stdout.txt')
    run('timing_pair.py',[],'timing_pair_stdout.txt')
    run('oracle_study.py',['--output','oracle_results_v2.json'],'oracle_stdout_v3.txt')
    run('exact_controls.py',[],'exact_controls_stdout.txt')
    run('local_nonlinear.py',[],'local_nonlinear_stdout.txt')
    run('large_covariance.py',[],'large_covariance_stdout.txt')
    run('make_report.py',[],'report_stdout.txt')
    run('verify_preservation.py',[],'preservation_stdout.txt')
if __name__=='__main__':main()
