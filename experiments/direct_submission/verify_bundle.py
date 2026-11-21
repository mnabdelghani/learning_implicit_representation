"""Check packaged hashes and small numerical interfaces; never run a full study."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true',help='Also check exact ODE/readout interfaces and replay 600 archived Burgers samples');args=parser.parse_args()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for rel,entry in manifest['files'].items():
        p=ROOT/rel
        if p.is_symlink() or not p.is_file():raise RuntimeError('Missing or linked package file: '+rel)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Hash mismatch: '+rel)
        if p.stat().st_size!=entry['bytes']:raise RuntimeError('Size mismatch: '+rel)
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(),filename=str(p.relative_to(ROOT)))
    checks={'hash_files_verified':len(manifest['files']),'python_sources_parsed':len(list(ROOT.rglob('*.py'))),'full_scientific_runs':0}
    if args.smoke:
        import numpy as np
        from scipy.integrate import solve_ivp
        direct=load('packaged_contact_direct_field',ROOT/'contact/direct_field.py')
        Q=direct.random_frame(3,8,81020);rng=np.random.default_rng(10820);A=rng.normal(size=(8,8));R=.001*A@A.T;v=rng.normal(size=8)*.03;dt=.01
        qpost,rpost,dg=direct.coupled_step(Q,R,v,dt,.5,100)
        def rhs(t,y):
            q=y[:24].reshape(3,8);r=y[24:].reshape(8,8);qr=q@r
            return np.r_[(-100*(qr-(qr@q.T)@q)).ravel(),(np.outer(v,v)-.5*r).ravel()]
        sol=solve_ivp(rhs,(0,dt),np.r_[Q.ravel(),R.ravel()],method='DOP853',rtol=2e-13,atol=2e-15)
        qerr=float(np.max(np.abs(qpost-sol.y[:24,-1].reshape(3,8))));rerr=float(np.max(np.abs(rpost-sol.y[24:,-1].reshape(8,8))))
        assert qerr<1e-8 and rerr<1e-12
        S=[0,1,2,3,5];C=[4,6,7];supplied=rng.normal(size=5);completed=direct.complete_rates(qpost,S,supplied,C);full=np.empty(8);full[S]=supplied;full[C]=completed;completion_error=float(np.linalg.norm(qpost@full));assert completion_error<1e-11
        checks['contact']={'coupled_q_error':qerr,'coupled_R_error':rerr,'same_field_completion_residual':completion_error}
        fourier=load('packaged_fourier_run',ROOT/'fourier/run.py');c=rng.normal(size=6);c/=np.linalg.norm(c);A=rng.normal(size=(6,6));M=.001*A@A.T;w=rng.normal(size=6)*.02
        cp,Mp=fourier.coupled_step(c,M,w,.02,8.,3.,1e-10,1e-12)
        def fode(t,y):
            q=y[:6];r=y[6:].reshape(6,6);return np.r_[-3*(r@q-(q@r@q)*q),(np.outer(w,w)-r/8).ravel()]
        fs=solve_ivp(fode,(0,.02),np.r_[c,M.ravel()],method='DOP853',rtol=2e-13,atol=2e-15);fe=float(np.max(np.abs(cp-fs.y[:6,-1])));assert fe<1e-8
        phase=.4;rate,valid=fourier.readout(cp,phase,.15,1e-6,1);assert valid;fr=float(abs(fourier.evaluate(cp,phase,1)@np.array([.15,rate])));assert fr<1e-12
        checks['fourier']={'coupled_normalized_row_error':fe,'same_field_readout_residual':fr}
        env=os.environ.copy()
        for var in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[var]='1'
        result=subprocess.run([sys.executable,str(ROOT/'burgers/verify_package.py')],cwd=ROOT/'burgers',env=env,text=True,capture_output=True,check=True)
        checks['burgers']=json.loads(result.stdout)
        checks['scope']='Small interface/ODE checks and archived-sample replay; no new cohort or experimental success claim'
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
