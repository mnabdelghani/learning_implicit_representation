"""Implicit feature-correlation learning; no derivative targets enter updates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

class ImplicitLearner:
    """Q=(theta,1), v=(scaled observed features, supplied contribution)."""
    def __init__(self, initial, scales, lower, upper, dt, memory, steps=8):
        self.scales=np.asarray(scales,float)
        self.theta=np.asarray(initial,float)/self.scales
        self.lower=np.asarray(lower,float)/self.scales
        self.upper=np.asarray(upper,float)/self.scales
        self.R=np.zeros((len(initial)+1,)*2)
        self.beta=np.exp(-dt/memory)
        self.steps=steps
        if np.any(self.lower>self.theta) or np.any(self.theta>self.upper):
            raise ValueError('Initial coefficients must be feasible')
    @property
    def coefficients(self): return self.theta*self.scales
    def channels(self, free, supplied):
        free=np.asarray(free,float)
        if free.ndim==1: free=free[None,:]
        return np.column_stack((free*self.scales,np.broadcast_to(supplied,(len(free),))))
    def observe(self, free, supplied, adapt=True):
        v=self.channels(free,supplied)
        q=np.r_[self.theta,1.]
        residual=v@q
        energy=float(np.mean(residual**2)/2)
        denom=np.sum(q*q)*np.sum(v*v,axis=1)
        active=denom>1e-20
        normalized=float(np.mean(residual[active]**2/denom[active])) if np.any(active) else np.nan
        self.R=self.beta*self.R+(1-self.beta)*(v.T@v/len(v))
        H=self.R[:-1,:-1]
        e0=float(q@self.R@q/2)
        if adapt:
            step=1/max(float(np.trace(H)),1e-12)
            for _ in range(self.steps):
                self.theta=np.clip(self.theta-step*(H@self.theta+self.R[:-1,-1]),self.lower,self.upper)
        q1=np.r_[self.theta,1.]
        return dict(energy=energy, normalized=normalized, feature_power=float(np.mean(np.sum(v*v,axis=1))),
                    memory_min_eigenvalue=float(np.linalg.eigvalsh(H)[0]), memory_before=e0,
                    memory_after=float(q1@self.R@q1/2))

def rms(x): return float(np.sqrt(np.mean(np.asarray(x)**2)))

def save_trace(path, rows):
    import csv
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dump_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def rowspace_error(F,G,coordinate_scales):
    """Frobenius distance between row projectors in declared scaled coordinates."""
    U=np.linalg.qr((np.asarray(F)*coordinate_scales).T,mode='reduced')[0]
    V=np.linalg.qr((np.asarray(G)*coordinate_scales).T,mode='reduced')[0]
    return float(np.linalg.norm(U@U.T-V@V.T))
