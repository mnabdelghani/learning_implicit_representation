"""Contact learning -> representation -> control, from one prespecified saved case.

No simulator/learner import, training, record mutation, or seed selection.
All 24 entries use the original evaluator-only orthogonal display alignment.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def causal_mean(values,count):
    cumulative=np.r_[0.,np.cumsum(values)]
    high=np.arange(len(values))+1;low=np.maximum(0,high-count)
    return (cumulative[high]-cumulative[low])/(high-low)

def main(case,output):
    output.mkdir(parents=True,exist_ok=False)
    (output/'plot_contact_source.py').write_bytes(Path(__file__).read_bytes())
    config=json.loads((case/'config.json').read_text())
    summary=json.loads((case/'summary.json').read_text())
    assert config['seed']==101 and summary['status']=='complete'
    raw=np.load(case/'states.npz')
    trace=np.genfromtxt(case/'trace.csv',delimiter=',',names=True)
    t=trace['time'];Q=raw['q_pre'];D=raw['scales'];A=raw['F_evaluator']*D
    # Same symmetric polar normalization and orthogonal Procrustes map as the
    # original contact report. Neither enters the learner or controller.
    values,vectors=np.linalg.eigh(A@A.transpose(0,2,1))
    assert np.all(values>0)
    inverse_root=(vectors*(1/np.sqrt(values))[:,None,:])@vectors.transpose(0,2,1)
    T=inverse_root@A
    left,overlap,right=np.linalg.svd(T@Q.transpose(0,2,1))
    O=left@right;aligned=O@Q;difference=aligned-T
    assert np.max(np.linalg.norm(O@O.transpose(0,2,1)-np.eye(3),axis=(1,2)))<1e-10
    reconstructed=np.linalg.norm(Q.transpose(0,2,1)@Q-T.transpose(0,2,1)@T,axis=(1,2))
    assert np.max(np.abs(reconstructed-trace['projector_error']))<1e-8
    pre_residual=(Q@(raw['v_observed']/D)[...,None])[...,0]
    energy_raw=.5*np.sum(pre_residual**2,axis=1)
    assert np.allclose(energy_raw,.5*trace['incompatibility']**2,rtol=1e-8,atol=1e-12)
    window=round(.5/config['dt'])
    energy=causal_mean(energy_raw,window)
    frozen_energy=causal_mean(.5*trace['frozen_incompatibility']**2,window)
    training_end=config['warmup'];change=config['change_time'];end=config['duration']
    control=t>=training_end;late=(t>=end-10)&(t<end)
    assert np.isclose(np.mean(trace['projector_error'][late]),summary['late']['projector_error'])
    maximum=float(np.max(np.abs(difference)))
    # One common symmetric color range includes every entry at every saved time.
    bound=float(np.ceil(maximum*10)/10)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.,'axes.titlesize':7.7,
        'axes.labelsize':7.,'legend.fontsize':6.2,'xtick.labelsize':6.3,'ytick.labelsize':6.3,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'lines.linewidth':1.0})
    fig=plt.figure(figsize=(7.3,4.25))
    grid=fig.add_gridspec(2,3,width_ratios=[1,1.25,1],left=.083,right=.985,
                          bottom=.18,top=.845,wspace=.54,hspace=.48)
    ax_energy=fig.add_subplot(grid[0,0]);ax_error=fig.add_subplot(grid[1,0])
    ax_entries=fig.add_subplot(grid[:,1]);ax_force=fig.add_subplot(grid[0,2]);ax_motion=fig.add_subplot(grid[1,2])
    blue='#126aa5';orange='#d66b20'
    for ax in [ax_energy,ax_error,ax_force,ax_motion]:
        ax.axvline(training_end,color='.45',lw=.8,ls=':')
        ax.axvline(change,color='.28',lw=.8,ls='--')
        ax.set_xlim(0,end);ax.set_xticks([0,40,60,120]);ax.set_xlabel('Time (s)')
        ax.grid(alpha=.15)
    thin=np.arange(0,len(t),5)
    ax_energy.semilogy(t[thin],energy[thin],color=blue,label='Learned')
    ax_energy.semilogy(t[thin],frozen_energy[thin],color=orange,label='Frozen initial')
    ax_energy.set_title('(a) Pre-update incompatibility',loc='left',pad=5)
    ax_energy.set_ylabel(r'$\frac{1}{2}\|Q D^{-1}\dot z\|^2$')
    ax_energy.legend(frameon=False,loc='lower left',handlelength=1.7)
    ax_energy.text(.98,.97,'0.5-s causal mean',transform=ax_energy.transAxes,ha='right',va='top',fontsize=6.)
    ax_error.plot(t[thin],trace['projector_error'][thin],color=blue)
    ax_error.plot(t[thin],trace['frozen_projector_error'][thin],color=orange)
    ax_error.set_title('(b) Differential row-space error',loc='left',pad=5)
    ax_error.set_ylabel(r'$\|\widehat P-P\|_F$');ax_error.set_ylim(0,2.5)
    ax_error.text(.98,.94,'Late error: '+f"{summary['late']['projector_error']:.3f}",transform=ax_error.transAxes,ha='right',va='top',fontsize=6.4)
    channels=[r'$x$',r'$y$',r'$\lambda$',r'$v_x$',r'$v_y$',r'$\dot\lambda$',r'$u_1$',r'$u_2$']
    # Feed every saved time/entry to the rasterized mesh. No time averaging,
    # entry-wise scaling, selection, or color saturation is applied.
    dt=float(config['dt']);edges=np.r_[t[0]-dt/2,(t[:-1]+t[1:])/2,t[-1]+dt/2]
    mesh=ax_entries.pcolormesh(edges,np.arange(25)-.5,difference.reshape(len(t),24).T,
                              cmap='RdBu_r',vmin=-bound,vmax=bound,shading='flat',rasterized=True)
    ax_entries.set_ylim(23.5,-.5);ax_entries.set_xlim(0,end)
    ax_entries.set_yticks(np.arange(24));ax_entries.set_yticklabels([f'{i+1}: '+channel for i in range(3) for channel in channels],fontsize=6.1)
    ax_entries.set_xticks([0,40,60,120]);ax_entries.set_xlabel('Time (s)')
    ax_entries.tick_params(axis='y',length=0,pad=3)
    for boundary in [7.5,15.5]:ax_entries.axhline(boundary,color='.28',lw=.8)
    ax_entries.axvline(training_end,color='.40',lw=.8,ls=':')
    ax_entries.axvline(change,color='.20',lw=.8,ls='--')
    ax_entries.set_title('(c) All 24 aligned-entry errors',loc='left',pad=5)
    ax_entries.text(-.02,1.005,'Row: channel',transform=ax_entries.transAxes,ha='right',va='bottom',fontsize=5.8)
    box=ax_entries.get_position()
    colorax=fig.add_axes([box.x0, .07, box.width, .022])
    colorbar=fig.colorbar(mesh,cax=colorax,orientation='horizontal',ticks=[-bound,0,bound])
    colorbar.ax.tick_params(labelsize=6.,length=2,pad=1)
    colorbar.ax.set_xlabel(r'Signed difference $OQ-T$ (display gauge)',fontsize=6.2,labelpad=2)
    ax_force.plot(t[thin],trace['lambda'][thin],color=blue,label='Actual')
    indices=np.flatnonzero(control)[::5]
    ax_force.plot(t[indices],trace['desired_lambda'][indices],color='.15',ls='--',label='Reference')
    ax_force.set_title('(d) Vertical-force control',loc='left',pad=5);ax_force.set_ylabel(r'$\lambda$ (N)')
    ax_force.legend(frameon=False,loc='best',ncol=1,handlelength=1.7)
    ax_motion.plot(t[thin],trace['x'][thin],color=blue)
    ax_motion.plot(t[indices],trace['desired_x'][indices],color='.15',ls='--')
    ax_motion.set_title('(e) Horizontal-motion control',loc='left',pad=5);ax_motion.set_ylabel(r'$x$ (m)')
    positions=[ax_energy.get_position(),ax_entries.get_position(),ax_force.get_position()]
    for box,label in zip(positions,['LEARN FROM RATES','REPRESENT THE RELATION','CONTROL THE TASK']):
        fig.text((box.x0+box.x1)/2,.954,label,ha='center',va='center',fontsize=7.5,fontweight='bold',color='#18374f')
    for start,endpos in [(positions[0].x1+.015,positions[1].x0-.02),(positions[1].x1+.02,positions[2].x0-.018)]:
        ax_entries.annotate('',xy=(endpos,.954),xytext=(start,.954),xycoords=fig.transFigure,
                            arrowprops=dict(arrowstyle='->',color='#577183',lw=1.0),annotation_clip=False)
    fig.legend([Line2D([],[],color='.45',ls=':',lw=.8),Line2D([],[],color='.28',ls='--',lw=.8)],
               ['40 s: control starts; learning continues','60 s: material/actuation change'],
               loc='upper center',bbox_to_anchor=(.53,.920),ncol=2,frameon=False,fontsize=6.6,handlelength=2.1,columnspacing=2.)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();canvas=fig.bbox
    outside=[]
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bounds=text.get_window_extent(renderer)
            if bounds.x0<canvas.x0-.5 or bounds.y0<canvas.y0-.5 or bounds.x1>canvas.x1+.5 or bounds.y1>canvas.y1+.5:
                outside.append(dict(text=text.get_text(),bounds=list(bounds.bounds)))
    for extension in ['pdf','png']:fig.savefig(output/f'direct_contact_summary.{extension}',dpi=240)
    plt.close(fig)
    np.savez_compressed(output/'display_data.npz',time_s=t,scales=D,truth_frame=T,alignment=O,aligned_Q=aligned,
        signed_entry_difference=difference,projector_error=trace['projector_error'],frozen_projector_error=trace['frozen_projector_error'],
        preupdate_energy=energy_raw,energy_causal_mean=energy,frozen_energy_causal_mean=frozen_energy,
        force=trace['lambda'],force_reference=trace['desired_lambda'],position=trace['x'],position_reference=trace['desired_x'])
    metadata=dict(illustration_seed=101,scientific_runs=0,source_record_mutations=0,
        input_sha256={name:sha(case/name) for name in ['states.npz','trace.csv','config.json','summary.json']},
        plotter_sha256=sha(Path(__file__)),figure_size_inches=[7.3,4.25],
        gauge='A=F_evaluator D; T=(A A^T)^(-1/2) A; U S V^T=SVD(T Q^T); O=U V^T; signed display difference=OQ-T. Alignment is evaluator-only; commands use original Q.',
        heatmap=dict(entries=24,saved_samples=len(t),rows='row-major 3x8',channels=['x','y','lambda','v_x','v_y','dot_lambda','u_1','u_2'],
            range=[-bound,bound],maximum_absolute_difference=maximum,color_mapping='common linear symmetric RdBu_r; no clipping, entry-wise scaling or time averaging; full saved array passed to rasterized pcolormesh'),
        curve_display_stride=5,energy_causal_mean_seconds=.5,energy_startup='shortened trailing window',
        markers=dict(control_start_s=training_end,plant_change_s=change),late_window_s=[end-10,end],late_right_endpoint_included=False,
        late_all24_entry_rms=np.sqrt(np.mean(difference[late]**2,axis=0)).tolist(),
        checks=dict(max_projector_reconstruction_error=float(np.max(np.abs(reconstructed-trace['projector_error']))),
                    max_alignment_orthogonality_error=float(np.max(np.linalg.norm(O@O.transpose(0,2,1)-np.eye(3),axis=(1,2)))),
                    energy_reconstruction_pass=True,canonical_material_normalization_used=False,text_outside_canvas=outside),
        outputs={path.name:dict(bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(output.iterdir()) if path.is_file()})
    (output/'figure_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),samples=len(t),color_range=[-bound,bound],maximum_entry_error=maximum,
                          late_projector_error=float(np.mean(reconstructed[late]))),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.case,args.output)
