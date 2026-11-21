"""Generate the supplementary table from the independently audited run metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new table output')
    data = json.loads(args.audit.read_text())
    if data['status'] != 'passed':
        raise ValueError('Table requires a successful independent audit')
    lines = [f'% Audit SHA-256: {hashlib.sha256(args.audit.read_bytes()).hexdigest()}',
             r'\begin{table*}[t]', r'\centering',
             r'\caption{New normalized Fourier protocol: mean $\pm$ sample standard deviation over seeds 10--19, using $60\le t<80$~s. Every numeric column is multiplied by $10^3$. $e_P$ is evaluated differential projector error; $e_F$ uses learned-input gauge alignment. Dashes mark shadows without their own control trajectory.}',
             r'\label{tab:normalized_fourier_results}',
             r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Stream & Estimator & $10^3 e_P$ & $10^3 e_F$ & $10^3$ phase RMSE & $10^3$ tracking RMSE \\',
             r'\midrule']
    metrics = ['derivative_projector_error_mean', 'aligned_F_error_mean',
               'phase_partial_rmse', 'tracking_rmse']
    for mode, mode_label in [('replay', 'Replay'), ('own_trajectory', 'Flow-controlled')]:
        for method, label in [('normalized_flow', 'Normalized flow'),
                              ('bottom_eigenvector', 'Memory minimum'), ('frozen_prior', 'Frozen prior')]:
            rows = [r for r in data['runs_data'] if r['mode'] == mode and r['method'] == method]
            if sorted(r['seed'] for r in rows) != list(range(10, 20)):
                raise ValueError('Incomplete declared seed set')
            cells = []
            for key in metrics:
                if key not in rows[0]['metrics']:
                    cells.append('---')
                    continue
                values = np.asarray([r['metrics'][key] for r in rows]) * 1000
                if not np.all(np.isfinite(values)):
                    raise ValueError('Do not silently exclude nonfinite outcomes')
                cells.append(f'${np.mean(values):.3f}\\pm{np.std(values, ddof=1):.3f}$')
            lines.append(' & '.join([mode_label, label, *cells]) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
