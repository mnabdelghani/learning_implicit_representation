"""Reproduce the fixed direct-contact confirmation cohort in a NEW directory."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,choices=[1,2],default=1);a=p.parse_args()
    protocol=json.loads((HERE/'confirmatory_protocol.json').read_text())
    for name,expected in protocol['sources'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen source mismatch: '+name)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'confirmatory_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    jobs=[(s,False) for s in protocol['confirmatory_seeds']]+[(protocol['refinement']['seed'],True)]
    def execute(job):
        seed,refine=job;name=f'{"refine" if refine else "confirm"}_seed{seed}';cmd=[sys.executable,str(HERE/'contact_direct.py'),'--output',str(a.output/name),'--seed',str(seed),'--duration','120','--change-time','60','--dt','.005' if refine else '.01']
        env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        with (a.output/(name+'_execution.log')).open('w') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        return dict(seed=seed,refinement=refine,returncode=result.returncode,directory=name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool: results=list(pool.map(execute,jobs))
    (a.output/'execution_jobs.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['returncode'] for r in results):raise SystemExit('Some runs failed; all records and execution logs retained.')
if __name__=='__main__':main()
