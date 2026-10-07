import json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;log=R/'remaining_stdout.txt';assert not log.exists()
commands=[['run_gates.py','--names','public_v29','centered','coherent'],['freeze_confirmation.py'],['screen.py','--names','baseline_snapshot','unclip34','centered','coherent','public_v25','public_v29','--reuse','--data',str(R/'public/confirmation_shard11.arrow'),'--ids','1','2','3','4','--output','confirmation.json'],['screen.py','--names','debug_baseline_snapshot','debug_unclip34','debug_centered','debug_coherent','--diagnostics','--output','diagnostic_states.json'],['diagnose_propagated.py']]
report=[]
with log.open('w') as stream:
    for args in commands:
        cmd=[sys.executable,str(R/args[0]),*args[1:]];print('Starting',args,flush=True);rc=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8')).returncode;stream.flush();report.append({'command':cmd,'returncode':rc});(R/'remaining_status.json').write_text(json.dumps(report,indent=2));print('Completed',args[0],rc,flush=True)
