"""Focused portable-source replay and literal-ODE checks, without complete reruns."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.integrate import solve_ivp
import burgers_direct as runner
import paired_burgers as direct
import native_kernel
H=Path(__file__).resolve().parent
F=H/'data/replay_seed732'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    checks={}
    for name,wanted in json.loads((F/'artifact_hashes.json').read_text()).items():assert sha(F/name)==wanted
    runner.verify_sources()
    cfg=json.loads((F/'config.json').read_text())
    for phase,stem in [('training','initial'),('adaptive','branch'),('frozen','branch')]:
        state=runner.load_checkpoint(cfg,F,stem)
        with np.load(F/(phase+'_chunk_00000.npz')) as saved:
            maximum=0.
            for i in range(len(saved['time'])):
                record,memory=runner.observe_and_step(state,cfg,phase=='training',phase!='frozen')
                for key in ['context','secant','q_pre','q_post','udot','udot_raw','supplied_rates','true_w']:
                    actual,expected=record[key],saved[key][i]
                    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12,equal_nan=True)
                    finite=np.isfinite(expected)
                    if finite.any():maximum=max(maximum,float(np.max(np.abs(actual[finite]-expected[finite]))))
            checks[phase+'_archived_samples_replayed']=len(saved['time'])
            checks[phase+'_max_abs_discrepancy']=maximum
    # A random observed-rate problem: same vector field, joint ODE, step refinement.
    rng=np.random.default_rng(93841);q=direct.unit_rows(rng.normal(size=(3,5)));A=rng.normal(size=(3,5,5));R=np.einsum('nij,nkj->nik',A,A)*.0003;v=rng.normal(size=(3,5))*.03;h=.005
    n=q.size
    def rhs(t,y):
        Q=y[:n].reshape(q.shape);M=y[n:].reshape(R.shape);QR=np.einsum('ni,nij->nj',Q,M)
        dQ=-10000*(QR-(QR*Q).sum(axis=1)[:,None]*Q);dM=v[:,:,None]*v[:,None,:]-M/4
        return np.r_[dQ.ravel(),dM.ravel()]
    exact=solve_ivp(rhs,(0,h),np.r_[q.ravel(),R.ravel()],method='DOP853',rtol=2e-13,atol=2e-15).y[:,-1]
    nativeQ,nativeR,_=native_kernel.native_step(q,R,v,h)
    qerr=np.max(np.abs(nativeQ-exact[:n].reshape(q.shape)));rerr=np.max(np.abs(nativeR-exact[n:].reshape(R.shape)))
    assert qerr<1e-7 and rerr<1e-12,(qerr,rerr)
    errors=[]
    for count in [1,2,4]:
        Q,M=q.copy(),R.copy()
        for k in range(count):Q,M,_=native_kernel.native_step(Q,M,v,h/count)
        errors.append(float(np.max(np.abs(Q-exact[:n].reshape(q.shape)))))
    assert errors[-1]<errors[0]
    checks.update(literal_joint_ode_q_error=float(qerr),literal_joint_ode_R_error=float(rerr),held_observation_step_refinement_errors=errors,
                  refinement_scope='Numerical held-observation learner test only; whole closed-loop dt sensitivity remains in historical audit',passed=True)
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
