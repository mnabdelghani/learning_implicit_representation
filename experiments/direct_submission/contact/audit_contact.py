"""Independent formula/record checks (same study author; external review still needed)."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
SOURCE_DIR=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--records',type=Path,default=SOURCE_DIR);args=parser.parse_args();HERE=args.records
sys.path.insert(0,str(SOURCE_DIR))
import contact_direct as impl
D=impl.D;S=impl.S;C=impl.C

def rms(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))
def projector(f):
 u=np.linalg.svd(f,full_matrices=False)[2][:3];return u.T@u

def recordcheck(folder,primary):
 cfg=json.loads((folder/'config.json').read_text());on=json.loads((folder/'summary.json').read_text())
 for name,sha in json.loads((folder/'manifest.json').read_text()).items():
  assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha,(folder,name)
 raw=np.load(folder/'states.npz');tr=np.genfromtxt(folder/'trace.csv',delimiter=',',names=True)
 q=raw['q_pre'];R=raw['R_pre'];z=raw['z'];v=raw['v_observed']/D;F=raw['F_evaluator'];q0=raw['q_initial'];t=tr['time']
 supplied_all=raw['supplied_query']
 P=np.array([projector(f*D) for f in F]);Pq=q.transpose(0,2,1)@q;P0=q0.T@q0
 metrics=dict(projector_error=np.linalg.norm(Pq-P,axis=(1,2)),frozen_projector_error=np.linalg.norm(P0-P,axis=(1,2)),incompatibility=np.linalg.norm(np.einsum('nmd,nd->nm',q,v),axis=1),frozen_incompatibility=np.linalg.norm(v@q0.T,axis=1),true_incompatibility=np.linalg.norm(np.einsum('nmd,nd->nm',F,raw['v_observed']),axis=1))
 differences={name:float(np.max(np.abs(vals-tr[name]))) for name,vals in metrics.items() if name!='true_incompatibility'}
 assert max(differences.values())<1e-10
 beta=np.exp(-cfg['dt']/cfg['tau']);R_expected=beta*R[:-1]+cfg['tau']*(1-beta)*v[:-1,:,None]*v[:-1,None,:]
 memdiff=float(np.max(np.abs(R_expected-R[1:])))
 assert memdiff<1e-10
 # Re-integrate chosen Q steps with many fixed RK4 substeps, exact evolving R.
 integration=[]
 for ix in [0,min(100,len(t)-2),min(3999,len(t)-2),min(6001,len(t)-2),len(t)-2]:
  Q=q[ix].copy();old=R[ix];outer=np.outer(v[ix],v[ix]);h=cfg['dt']/200
  def rhs(s,x):
   cov=np.exp(-s/cfg['tau'])*old+cfg['tau']*(-np.expm1(-s/cfg['tau']))*outer
   qr=x@cov;return -cfg['gain']*(qr-(qr@x.T)@x)
  for j in range(200):
   s=j*h;k1=rhs(s,Q);k2=rhs(s+h/2,Q+h*k1/2);k3=rhs(s+h/2,Q+h*k2/2);k4=rhs(s+h,Q+h*k3);Q+=h*(k1+2*k2+2*k3+k4)/6
  integration.append(dict(index=int(ix),Q_next_difference=float(np.linalg.norm(Q-q[ix+1]))))
 assert max(x['Q_next_difference'] for x in integration)<1e-7
 # Evaluator finite-difference check, calculated independently of sensor.
 finite=[];sim=impl.Simulator(cfg['change_time'])
 for ix in [0,len(t)//3,len(t)*2//3,len(t)-1]:
  a,b,B=sim.coefficients(t[ix]);g=np.array([.035,.020]);zz=z[ix];w=np.random.default_rng(ix).normal(size=8)
  def residual(zz):
   x,y,l,vx,vy,ld,u1,u2=zz;h=g@np.array([np.sin(2*x),np.cos(5*x)]);hp=g@np.array([2*np.cos(2*x),-5*np.sin(5*x)])
   return np.r_[ld-a*(h-y)+4*l-b*(hp*vx-vy),np.array([vx,vy])-B@np.array([u1,u2])]
  eps=1e-6;fd=(residual(zz+eps*w)-residual(zz-eps*w))/(2*eps);finite.append(float(np.linalg.norm(fd-F[ix]@w)/max(1,np.linalg.norm(F[ix]@w))))
 assert max(finite)<1e-7
 late=(t>=110)&(t<120);entries=np.max(np.abs(q-q0),axis=0)
 result=dict(seed=cfg['seed'],status=on['status'],samples=len(t),source_hashes=json.loads((folder/'environment.json').read_text())['source_hashes'],metric_max_abs_differences=differences,memory_max_abs_difference=memdiff,selected_RK4_checks=integration,max_directional_difference=max(finite),max_true_tangent_residual=float(np.max(metrics['true_incompatibility'])),max_row_orthogonality=float(np.max(np.linalg.norm(q@q.transpose(0,2,1)-np.eye(3),axis=(1,2)))),minimum_entry_max_change=float(entries.min()),all24_entries_changed=bool(np.all(entries>1e-5)))
 if late.any():
  query=[];frozen=[];controllererrors=[];errors=[]
  for i in np.flatnonzero(t>=40):
   supplied=supplied_all[i]
   if not np.isfinite(supplied).all():continue
   qt=q[i];truth=np.linalg.svd(F[i]*D,full_matrices=False)[2][:3]
   if np.linalg.svd(qt[:,C],compute_uv=False)[-1]<cfg['guard']:continue
   completion=np.linalg.solve(qt[:,C],-qt[:,S]@(supplied/D[S]))*D[C]
   expected=np.linalg.solve(truth[:,C],-truth[:,S]@(supplied/D[S]))*D[C]
   err=np.linalg.norm(completion-expected);errors.append(abs(err-tr['query_error'][i]))
   limited=np.clip(completion[1:],-cfg['input_rate_limit'],cfg['input_rate_limit']);u=tr[['u1','u2']][i].tolist();u=np.array(u)
   actual=(np.clip(u+cfg['dt']*limited,-cfg['input_limit'],cfg['input_limit'])-u)/cfg['dt'];controllererrors.append(np.max(np.abs(actual-np.array([tr['udot1'][i],tr['udot2'][i]]))))
  assert max(errors)<1e-10 and max(controllererrors)<1e-10
  result.update(query_metric_max_abs_difference=float(max(errors)),same_Q_control_max_abs_difference=float(max(controllererrors)),late=dict(x_rmse=rms(tr['x'][late]-tr['desired_x'][late]),lambda_rmse=rms(tr['lambda'][late]-tr['desired_lambda'][late]),projector_mean=float(metrics['projector_error'][late].mean()),projector_ratio=float(metrics['projector_error'][late].mean()/metrics['frozen_projector_error'][late].mean()),incompatibility_ratio=float(metrics['incompatibility'][late].mean()/metrics['frozen_incompatibility'][late].mean()),query_rms=rms(tr['query_error'][late]),frozen_query_rms=rms(tr['frozen_query_error'][late]),query_ratio=rms(tr['query_error'][late])/rms(tr['frozen_query_error'][late]),saturation_count=int(tr['saturated'][t>=40].sum()),guard_count=int(tr['guarded'][t>=40].sum()),min_indentation=float(tr['indentation'].min()),min_control_sigma=float(tr['readout_sigma'][t>=40].min())))
  m=result['late'];result['predeclared_acceptance']=bool(on['status']=='complete' and len(t)==round(cfg['duration']/cfg['dt'])+1 and m['x_rmse']<.01 and m['lambda_rmse']<.2 and m['projector_mean']<.6 and m['projector_ratio']<.5 and m['incompatibility_ratio']<.5 and m['query_ratio']<.5 and m['saturation_count']==0 and m['guard_count']==0)
 return result

out=dict(status='checks_passed',independence='Recomputed from raw records with independent formulas; author external audit still required',runs=[])
for path in sorted(HERE.glob('confirm_seed*'))+sorted(HERE.glob('refine_seed*')):
 if path.is_dir():out['runs'].append(recordcheck(path,True))
coarse=np.load(HERE/'confirm_seed101/states.npz');fine=np.load(HERE/'refine_seed101/states.npz');tc=np.genfromtxt(HERE/'confirm_seed101/trace.csv',delimiter=',',names=True);tf=np.genfromtxt(HERE/'refine_seed101/trace.csv',delimiter=',',names=True)
assert np.allclose(tc['time'],tf['time'][::2]);window=(tc['time']>=110)&(tc['time']<120)
out['refinement']=dict(same_time_late_state_rmse=np.sqrt(np.mean((coarse['z'][window,:3]-fine['z'][::2][window,:3])**2,axis=0)).tolist(),same_time_late_projector_rmse=rms(tc['projector_error'][window]-tf['projector_error'][::2][window]),coarse_summary=json.loads((HERE/'confirm_seed101/summary.json').read_text())['late'],fine_summary=json.loads((HERE/'refine_seed101/summary.json').read_text())['late'])
(HERE/'audit_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
