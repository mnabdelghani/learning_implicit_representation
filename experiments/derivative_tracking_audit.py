"""Independent postprocessing: evaluated-partial error excluding supplied blocks.

Uses saved pre-update partials/coefficients only. No simulation or fitting runs.
The denominator excludes entries supplied exactly to the learner, so fixed known
blocks cannot dilute the reported error. Each system retains its declared units.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = {"contact": ["learned", "frozen", "oracle"], "fourier": ["learned", "K0", "frozen", "oracle"], "burgers": ["learned", "frozen", "oracle"]}
WINDOWS = {"contact": {"early": [0,20], "before": [40,60], "change": [60,80], "late": [100,120]}, "fourier": {"early": [0,20], "late": [60,80]}, "burgers": {"early": [0,6], "late": [18,24]}}


def latex_number(value, digits):
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return mantissa + r"\times10^{" + str(int(exponent)) + "}"
    return text


def relative_error(estimate, truth, mask):
    a, b = np.asarray(estimate)[:, mask], np.asarray(truth)[:, mask]
    numerator = np.linalg.norm(a-b, axis=1)
    denominator = np.linalg.norm(b, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Undefined relative error; report the degenerate case explicitly")
    return numerator / denominator


def run(records, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new diagnostic output directory")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    (output / source.name).write_bytes(source.read_bytes())
    (output / "provenance.json").write_text(json.dumps(dict(
        command=sys.argv, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, matplotlib=matplotlib.__version__,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        output_identity="derivative_tracking_20261118_final",
        purpose="Postprocessing unchanged reproduced records; no simulation or fitting",
    ), indent=2) + "\n")
    results, curves, hashes = [], {}, {}
    for study, methods in METHODS.items():
        cfg_path = records / study / "config.json"
        cfg = json.loads(cfg_path.read_text())
        hashes[str(cfg_path.relative_to(records))] = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        for method in methods:
            curves[study, method] = []
            for seed in range(10,20):
                folder = records / study / f"seed_{seed}" / method
                trace_path = folder / "trace.csv"
                trace = np.genfromtxt(trace_path, delimiter=",", names=True)
                hashes[str(trace_path.relative_to(records))] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
                if study == "fourier":
                    t = trace["time"]
                    phase = cfg["phase_rate"] * t
                    basis = np.column_stack((np.ones(len(t)), np.cos(phase), np.sin(phase)))
                    dbasis = np.column_stack((np.zeros(len(t)), -np.sin(phase), np.cos(phase)))
                    c = np.column_stack([trace[f"c{k}"] for k in range(6)])
                    actual = np.broadcast_to(np.array(cfg["true_coefficients"]), c.shape)
                    rho = np.sum(actual[:,:3] * basis, axis=1)
                    kappa = np.sum(actual[:,3:] * basis, axis=1)
                    rate = (trace["control"] - kappa * trace["state"]) / rho
                    def F(coeff):
                        return np.column_stack((np.sum(coeff[:,3:]*basis,axis=1), np.sum(coeff[:,:3]*basis,axis=1), -np.ones(len(t)), np.sum(coeff[:,:3]*dbasis,axis=1)*rate+np.sum(coeff[:,3:]*dbasis,axis=1)*trace["state"])) * np.array(cfg["context_scales"])
                    true, est = F(actual), F(c)
                    mask = np.array([True,True,False,True])
                    expected_full = trace["F_error"]
                else:
                    partials_path = folder / "partials.npz"
                    hashes[str(partials_path.relative_to(records))] = hashlib.sha256(partials_path.read_bytes()).hexdigest()
                    with np.load(partials_path) as data:
                        t = data["time"].copy(); true=data["F_true"].copy(); est=data["F_est"].copy()
                    if study == "contact":
                        scale = np.array(cfg["row_scales"])[:,None] * np.array(cfg["context_scales"])
                        true *= scale; est *= scale
                        mask2 = np.zeros((3,8),bool); mask2[0,[0,1,3,4]]=True; mask2[1:,6:]=True
                    else:
                        mask2 = np.ones(true.shape[1:],bool); mask2[...,-1]=False
                    true, est, mask = true.reshape(len(t),-1), est.reshape(len(t),-1), mask2.ravel()
                    ix = np.searchsorted(trace["time"],t)
                    np.testing.assert_allclose(trace["time"][ix],t,atol=1e-12,rtol=0)
                    expected_full = trace["F_error" if study=="contact" else "F_relative_error_preupdate"][ix]
                assert np.array_equal(est[:,~mask],true[:,~mask]), (study,method,seed,"fixed blocks changed")
                full = relative_error(est,true,np.ones(est.shape[1],bool))
                np.testing.assert_allclose(full,expected_full,atol=1e-10,rtol=1e-8)
                unknown = relative_error(est,true,mask)
                metrics = {}
                for window,(lo,hi) in WINDOWS[study].items():
                    use = (t>=lo)&(t<hi)
                    metrics[window] = dict(unknown_only_mean=float(unknown[use].mean()),full_mean=float(full[use].mean()),samples=int(use.sum()))
                results.append(dict(study=study,method=method,seed=seed,metrics=metrics))
                curves[study,method].append((t,unknown))
    aggregate = {}
    for study,methods in METHODS.items():
        aggregate[study] = {}
        for method in methods:
            entries=[r for r in results if r["study"]==study and r["method"]==method]
            aggregate[study][method]={}
            for win in WINDOWS[study]:
                aggregate[study][method][win]={}
                for key in ["unknown_only_mean","full_mean"]:
                    a=np.array([r["metrics"][win][key] for r in entries])
                    aggregate[study][method][win][key]=dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),n=len(a))
    meta=dict(protocol="Postprocessing only, all 100 reproduced main-protocol runs",definition="Mean per-snapshot relative Frobenius error over learned entries; supplied entries excluded from numerator and denominator",windows=WINDOWS,all_results=results,aggregate=aggregate,source_sha256=hashes,limitations="Contact and Burgers use saved partial snapshots; Fourier uses all state records. These are online evaluated-partial errors on each method's own trajectory, not independent held-out data or cross-system comparable units.")
    (output/"metrics.json").write_text(json.dumps(meta,indent=2)+"\n")
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.3),constrained_layout=True)
    for ax,study in zip(axes,METHODS):
        for method,color in [("learned","#0068a6"),("frozen","#b95813")]:
            entries=curves[study,method]; t=entries[0][0]; data=np.array([x[1] for x in entries])
            mean=data.mean(axis=0); sd=data.std(axis=0,ddof=1)
            ax.plot(t,mean,label=method,color=color,lw=1.1)
            ax.fill_between(t,np.maximum(mean-sd,1e-9),np.maximum(mean+sd,1e-9),color=color,alpha=.18)
        if study=="contact": ax.axvline(60,color=".4",ls=":",lw=.8)
        ax.set(title=study.capitalize(),xlabel="Time (s)",yscale="log",ylim=(1e-6,3))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel("Relative error: learned partial entries")
    fig.savefig(output/"unknown_partial_tracking.pdf")
    fig.savefig(output/"unknown_partial_tracking.png",dpi=160)
    plt.close(fig)
    lines=[r"\begin{tabular}{llcc}",r"\toprule Study & Method & All entries & Learned entries \\",r"\midrule"]
    for study,methods in METHODS.items():
        for method in methods:
            a=aggregate[study][method]["late"]
            fmt=lambda x: "$"+latex_number(x['mean'],3)+r"\pm"+latex_number(x['sd'],2)+"$"
            lines.append(f"{study.capitalize()} & {method} & {fmt(a['full_mean'])} & {fmt(a['unknown_only_mean'])} \\\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (output/"unknown_partial_table.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps({"runs":len(results),"aggregate":aggregate},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--records",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();run(a.records,a.output)
