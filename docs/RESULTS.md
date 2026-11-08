# Executed results

Ten noise seeds (10–19), all included. Standard deviations are sample standard deviations.
Contact uses [100,120) s, Fourier [60,80) s, Burgers [18,24) s.

| Study | Method | Task RMS (mean ± SD) | Mean scaled differential error |
|---|---|---:|---:|
| contact | learned | 0.003424863 ± 0.000181 | 0.000895904 ± 0.000433 |
| contact | frozen | 0.8554918 ± 0.000275 | 0.583578 ± 3.2e-08 |
| contact | oracle | 0.003234699 ± 0.000133 | 0 ± 0 |
| fourier | learned | 0.002298958 ± 3.12e-05 | 0.00687125 ± 0.000188 |
| fourier | K0 | 0.04792815 ± 1.13e-05 | 0.184933 ± 8.61e-06 |
| fourier | frozen | 0.06727334 ± 0 | 0.214235 ± 2.93e-17 |
| fourier | oracle | 0.001228617 ± 0 | 0 ± 0 |
| burgers | learned | 0.0001974733 ± 1.94e-06 | 0.000107351 ± 1.08e-05 |
| burgers | frozen | 0.01658457 ± 3.66e-18 | 0.158437 ± 2.93e-17 |
| burgers | oracle | 0.0001976558 ± 2.86e-20 | 0 ± 0 |
