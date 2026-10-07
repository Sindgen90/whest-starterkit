"""Continue this active experiment serially after the first controller completes."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;log=R/'final_stdout.txt';assert not log.exists()
while True:
    p=R/'remaining_status.json'
    if p.exists():
        try:status=json.loads(p.read_text())
        except json.JSONDecodeError:status=[]
        if len(status)==5:assert all(x['returncode']==0 for x in status);break
    time.sleep(2)
commands=[['screen.py','--names','debug_public_v29','--reuse','--diagnostics','--output','public_diagnostic_states.json'],['diagnose_public.py'],['run_gates.py','--names','public_v29','--n-mlps','4','--seed','20261171','--suffix','confirmation_gate']]
report=[]
with log.open('w') as out:
    for args in commands:
        cmd=[sys.executable,str(R/args[0]),*args[1:]];print('Starting',args,flush=True);rc=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8')).returncode;out.flush();report.append({'command':cmd,'returncode':rc});(R/'final_status.json').write_text(json.dumps(report,indent=2));print('Completed',args[0],rc,flush=True)
