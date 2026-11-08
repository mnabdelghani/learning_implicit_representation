"""Aggregate every declared seed; populate manuscript values from saved run results."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def stats(values):
    a=np.asarray(values,float)
    if len(a)!=10:raise RuntimeError(f'Expected 10 completed seeds, got {len(a)}')
    if not np.all(np.isfinite(a)):raise ValueError('Nonfinite reported metric')
    return dict(mean=float(np.mean(a)),sd=float(np.std(a,ddof=1)),n=len(a))

def tex_stat(s):
    m,d=s['mean'],s['sd']
    if abs(m)<1e-15:return '0'
    e=int(np.floor(np.log10(abs(m))))
    if e<-2 or e>2:
        if d<1e-14:return f'{m/10**e:.3f}\\times10^{{{e}}}'
        return f'({m/10**e:.3f}\\pm{d/10**e:.3f})\\times10^{{{e}}}'
    if d<1e-14:return f'{m:.5f}'
    return f'{m:.5f}\\pm{d:.5f}'

def main():
    agg={}
    for ex,methods in [('contact',['learned','frozen','oracle']),('fourier',['learned','K0','frozen','oracle']),('burgers',['learned','frozen','oracle'])]:
        runs=json.loads((ROOT/'results'/ex/'summary.json').read_text())['runs'];out={}
        for method in methods:
            subset=[r for r in runs if r['method']==method]
            if sorted(r['seed'] for r in subset)!=list(range(10,20)):raise ValueError(f'Missing or duplicated seeds for {ex} {method}')
            if ex=='contact':
                keys=subset[0]['metrics']['recovered'].keys()
                out[method]={key:stats([r['metrics']['recovered'][key] for r in subset]) for key in keys}
                out[method]['minimum_indentation']=min(r['metrics']['minimum_indentation'] for r in subset)
                out[method]['max_saturation_fraction']=max(r['metrics']['saturation_fraction'] for r in subset)
                out[method]['change_force_rmse']=stats([r['metrics']['change']['force_rmse'] for r in subset])
            else:
                out[method]={key:stats([r['metrics'][key] for r in subset]) for key in subset[0]['metrics'] if np.isscalar(subset[0]['metrics'][key])}
        agg[ex]=out
    (ROOT/'results/aggregate.json').write_text(json.dumps(agg,indent=2)+'\n')
    macros={
     'ContactForce':('contact','learned','force_rmse'),'ContactMotion':('contact','learned','motion_rmse'),
     'ContactFError':('contact','learned','F_error_mean'),'ContactFrozen':('contact','frozen','force_rmse'),
     'ContactOracle':('contact','oracle','force_rmse'),
     'FourierShadowEnergy':('fourier','learned','K0_same_stream_energy_mean'),
     'FourierEnergy':('fourier','learned','energy_mean'),'FourierOracleEnergy':('fourier','learned','oracle_energy_mean'),
     'FourierTask':('fourier','learned','tracking_rmse'),'FourierKZeroTask':('fourier','K0','tracking_rmse'),
     'FourierOracleTask':('fourier','oracle','tracking_rmse'),
     'BurgersTask':('burgers','learned','tracking_rms'),'BurgersFrozen':('burgers','frozen','tracking_rms'),
     'BurgersOracle':('burgers','oracle','tracking_rms'),'BurgersFError':('burgers','learned','F_relative_error_mean')}
    tex='% GENERATED from 100 executed runs; do not hand-edit results.\n'
    for name,(ex,m,k) in macros.items():tex+=f'\\newcommand{{\\{name}}}{{{tex_stat(agg[ex][m][k])}}}\n'
    tex+=f'\\newcommand{{\\ContactSaturation}}{{{agg["contact"]["learned"]["max_saturation_fraction"]:.6g}}}\n'
    (ROOT/'supplement/results_macros.tex').write_text(tex)
    lines=['# Executed results','', 'Ten noise seeds (10–19), all included. Standard deviations are sample standard deviations.',
       'Contact uses [100,120) s, Fourier [60,80) s, Burgers [18,24) s.','',
       '| Study | Method | Task RMS (mean ± SD) | Mean scaled differential error |',
       '|---|---|---:|---:|']
    for ex,methods in [('contact',['learned','frozen','oracle']),('fourier',['learned','K0','frozen','oracle']),('burgers',['learned','frozen','oracle'])]:
        for m in methods:
            a=agg[ex][m];key={'contact':'force_rmse','fourier':'tracking_rmse','burgers':'tracking_rms'}[ex]
            fk='F_relative_error_mean' if ex=='burgers' else 'F_error_mean'
            t=a[key];f=a[fk]
            lines.append(f'| {ex} | {m} | {t["mean"]:.7g} ± {t["sd"]:.3g} | {f["mean"]:.6g} ± {f["sd"]:.3g} |')
    (ROOT/'docs/RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('\n'+tex)
if __name__=='__main__':main()
