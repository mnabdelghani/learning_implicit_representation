"""Replot retained airfoil fields; no physical simulation is executed.

Example (run from repository root, with existing research environment):
  ../.venv/bin/python experiments/plot_active_flow_fields.py --output ../notes/s8_flow_20261120/preview

The saved adaptive velocity gives vorticity by the original Fourier curl and
instantaneous streamlines by spatial interpolation. These legacy records use
structured coefficient learning, not the later direct-partial protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy import fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
VIEW = (1.15, 4.70, 1.05, 2.05)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(cfg):
    """Nominal closed-edge NACA profile; geometry only, not a fitted boundary."""
    q = (1 - np.cos(np.linspace(0, np.pi, 501))) / 2
    h = 5 * cfg['thickness_ratio'] * (0.2969*np.sqrt(q) - 0.1260*q
        - 0.3516*q**2 + 0.2843*q**3 - 0.1036*q**4)
    xx = np.r_[q, q[-2::-1]] - 0.5
    yy = np.r_[h, -h[-2::-1]]
    angle = np.deg2rad(cfg['angle_deg'])
    return np.c_[cfg['airfoil_center_x'] + cfg['chord'] * (np.cos(angle)*xx - np.sin(angle)*yy),
                 cfg['airfoil_center_y'] + cfg['chord'] * (np.sin(angle)*xx + np.cos(angle)*yy)]


def decorate(ax, x, y, data, cfg, polygon, zoom=False):
    c = cfg['chord']
    ax.add_patch(Polygon(polygon/c, closed=True, facecolor='white', edgecolor='#262c33',
                         linewidth=1.2, zorder=8))
    ax.contour(x/c, y/c, data['mask'], levels=[0.5], colors=['#65727e'],
               linewidths=0.95, linestyles=':', zorder=9)
    ax.contour(x/c, y/c, data['readout_weight'],
               levels=[0.5*float(data['readout_weight'].max())], colors=['#76538a'],
               linewidths=1.05, linestyles='--', zorder=9)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$y/c$')
    if zoom:
        ax.set_xlim(VIEW[:2]); ax.set_ylim(VIEW[2:])
    else:
        ax.set_xlim(0, cfg['lx']/c); ax.set_ylim(0, cfg['ly']/c)
        ax.add_patch(Rectangle((VIEW[0],VIEW[2]),VIEW[1]-VIEW[0],VIEW[3]-VIEW[2],
                              fill=False,edgecolor='#6b737b',linewidth=0.8,linestyle=(0,(5,4)),zorder=10))
    ax.tick_params(length=3)
    for s in ax.spines.values(): s.set_color('#6b737b'); s.set_linewidth(.65)


def figure_axes(title):
    fig = plt.figure(figsize=(10.8, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1,.025], height_ratios=[1,1.08],
                          left=.08,right=.93,bottom=.10,top=.88,hspace=.39,wspace=.045)
    axes = [fig.add_subplot(gs[i,0]) for i in range(2)]
    cbax = fig.add_subplot(gs[:,1])
    fig.suptitle(title, fontsize=14, x=.49, y=.975)
    handles=[Line2D([],[],color='#262c33',lw=1.2,label='Nominal NACA 0012 profile'),
             Line2D([],[],color='#65727e',ls=':',lw=1,label=r'Saved mask $\chi=0.5$'),
             Line2D([],[],color='#76538a',ls='--',lw=1,label='Half-maximum wake readout')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.49,.94),ncol=3,frameon=False,fontsize=11.0)
    return fig,axes,cbax


def save(fig, output, name):
    fig.savefig(output/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output/f'{name}.png',dpi=200)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'examples/legacy_suite/results/airfoil')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    names=['active_flow_vorticity','active_flow_streamlines']
    if any((args.output/(name+ext)).exists() for name in names for ext in ['.pdf','.png']):
        raise FileExistsError('Use a fresh output directory; existing figures are not overwritten.')
    summary=json.loads((args.data/'summary.json').read_text());cfg=summary['config']
    with np.load(args.data/'fields.npz',allow_pickle=False) as raw:
        data={k:raw[k] for k in raw.files}
    x,y=data['x'],data['y'];u,v=data['final_adaptive'];c=cfg['chord'];U=1.0
    if u.shape!=(cfg['ny'],cfg['nx']) or not np.all(np.isfinite([u,v])):
        raise ValueError('Unexpected saved field.')
    np.testing.assert_allclose(x,np.arange(cfg['nx'])*cfg['lx']/cfg['nx'],rtol=0,atol=1e-13)
    np.testing.assert_allclose(y,np.arange(cfg['ny'])*cfg['ly']/cfg['ny'],rtol=0,atol=1e-13)
    kx=2*np.pi*fft.rfftfreq(cfg['nx'],d=cfg['lx']/cfg['nx'])
    ky=2*np.pi*fft.fftfreq(cfg['ny'],d=cfg['ly']/cfg['ny'])
    omega=fft.irfft2(1j*(kx[None,:]*fft.rfft2(v)-ky[:,None]*fft.rfft2(u)),s=u.shape)
    vort=omega*c/U;speed=np.hypot(u,v)/U
    polygon=profile(cfg)
    X,Y=np.meshgrid(x,y)
    interior=MplPath(polygon).contains_points(np.c_[X.ravel(),Y.ravel()]).reshape(u.shape)
    stream_mask=interior|(data['mask']>=.5)
    plt.rcParams.update({'font.size':11,'axes.titlesize':11.5,'axes.labelsize':12,
                         'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42})
    # Symmetric full-data range: no percentile clipping or data smoothing.
    limit=float(np.max(np.abs(vort)))
    cmap=LinearSegmentedColormap.from_list('light_signed',['#5596bf','#c4dfed','#fffdf8','#f6d1ad','#dd875a'])
    fig,axs,cax=figure_axes('Final online-control vorticity  |  control time $tU_\\infty/c=6$')
    for i,ax in enumerate(axs):
        im=ax.pcolormesh(x/c,y/c,vort,cmap=cmap,norm=Normalize(-limit,limit),shading='nearest',rasterized=True)
        decorate(ax,x,y,data,cfg,polygon,zoom=(i==1))
        if i == 0:
            ax.text(.015,.975,'(a) Full retained domain',transform=ax.transAxes,va='top',fontsize=11.5,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':2},zorder=11)
        else:
            ax.set_title('(b) Enlarged wing and wake',loc='left',pad=7)
    fig.colorbar(im,cax=cax,label=r'Vorticity $\omega c/U_\infty$ (linear, full range)',ticks=[-8,-4,0,4,8])
    fig.text(.08,.025,'Same saved seed-10 snapshot in both views; no spatial smoothing. White profile is a geometry overlay.',fontsize=11.0,color='#4c535c')
    save(fig,args.output,names[0])
    speedmap=LinearSegmentedColormap.from_list('light_speed',['#fffdf5','#f6e4ac','#b9d8d6','#6baab7'])
    fig,axs,cax=figure_axes('Instantaneous online-control streamlines  |  control time $tU_\\infty/c=6$')
    uu=np.ma.array(u/U,mask=stream_mask);vv=np.ma.array(v/U,mask=stream_mask)
    for i,ax in enumerate(axs):
        im=ax.pcolormesh(x/c,y/c,speed,cmap=speedmap,norm=Normalize(0,float(speed.max())),shading='nearest',rasterized=True)
        ax.streamplot(x/c,y/c,uu,vv,color='#3e5260',density=(2.0,1.25),linewidth=.7,
                      arrowsize=.75,minlength=.08,maxlength=5,integration_direction='both',zorder=5)
        decorate(ax,x,y,data,cfg,polygon,zoom=(i==1))
        if i == 0:
            ax.text(.015,.975,'(a) Full retained domain',transform=ax.transAxes,va='top',fontsize=11.5,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':2},zorder=11)
        else:
            ax.set_title('(b) Enlarged wing and wake',loc='left',pad=7)
    fig.colorbar(im,cax=cax,label=r'Speed $|\mathbf{v}|/U_\infty$ (linear, full range)',ticks=np.arange(0,1.21,.2))
    fig.text(.08,.025,'Lines follow the instantaneous saved velocity, not particle paths. Interior lines are masked for visibility.',fontsize=11.0,color='#4c535c')
    save(fig,args.output,names[1])
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [args.data/'fields.npz',args.data/'summary.json',args.data/'trace.csv',Path(__file__)]}
    provenance={'study':'Retained legacy airfoil seed 10; visualization only','new_simulation_executed':False,
       'source_branch':'final_adaptive','control_time':cfg['control_time'],'total_simulation_time':sum(cfg[k] for k in ['spinup_time','training_time','control_time']),
       'grid_shape_yx':list(u.shape),'inputs_sha256':inputs,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
       'vorticity':'irfft2(i*(kx*rfft2(v_y)-ky*rfft2(v_x))), exactly the saved solver convention; displayed omega*c/Uinf, c=1 and Uinf=1',
       'vorticity_range':[float(vort.min()),float(vort.max())],'symmetric_color_range':[-limit,limit],'vorticity_colormap':['#5596bf','#c4dfed','#fffdf8','#f6d1ad','#dd875a'],
       'speed_range':[float(speed.min()),float(speed.max())],'speed_color_range':[0,float(speed.max())],
       'plotting':'Native-grid pcolormesh shading nearest; no filtering, resampling, percentile clipping, or smoothing of raster fields; half-cell support clipped to domain axes. Shared scale between full and zoom views.',
       'full_domain_chords':[0,cfg['lx']/c,0,cfg['ly']/c],'zoom_chords':list(VIEW),
       'geometry':'Nominal analytic closed-trailing-edge NACA0012 profile from recorded config, white-filled overlay. Saved mask chi=0.5 separately dotted; these differ on the coarse grid.',
       'streamlines':'Matplotlib streamplot using bilinear interpolation of saved final_adaptive velocity; automatic uniform seeding density=(2,1.25), both directions, arrowsize=.75, minlength=.08, maxlength=5. Same full-domain lines shown in both views. Not pathlines.',
       'stream_mask':'Grid points inside nominal polygon OR chi>=0.5; 2D interpolation stops at masks; no no-penetration condition imposed on stored field. Geometry overlay hides body interior in both plots.',
       'masked_stream_points':int(stream_mask.sum()),'readout_outline':'Saved readout_weight at half its grid maximum',
       'limitations':'Legacy structured-coefficient learner; not direct partial learning. One retained final snapshot; coarse immersed geometry, nonzero immersed speed, no aerodynamic or grid/time/penalty validation. Masking is only a display choice.',
       'outputs_sha256':{p.name:sha(p) for name in names for ext in ['.pdf','.png'] for p in [args.output/(name+ext)]}}
    (args.output/'active_flow_fields_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'vorticity_range':provenance['vorticity_range'],'speed_range':provenance['speed_range'],'masked_points':int(stream_mask.sum())},indent=2))

if __name__=='__main__': main()
