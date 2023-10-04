# Test data for neural-network preconditioners for Ref. [1]
This repository contains all of the test data presented in Ref. [1] of two-dimensional, two-flavor $U(1)$ lattice gauge theories. ***Angles*** of $U(1)$ fields are stored in `npy` files. $U(1)$ fields can be easily loaded by, for example,
```python
import numpy as np
file1 = "config.l16-N200-b2.0-k0.276-unquenched-test.x.npy"
U1_field = np.exp(1j * np.load(files))
```

All configuration files follow the naming format `config.l{lattice_geom}-N{num_config}-b{beta}-k{kappa}-unquenched-test.x.npy` where
  - `lattice_geom`: lattice dimension (all lattices have equal dimensions in all directions).
  - `num_config`: number of configurations.
  - `beta`: $\beta$, the inverse gauge coupling.
  - `kappa`: $\kappa$, the hopping parameter. It is related to the quark mass $m$ by $\kappa = 1/(2(m+2))$.
See Ref. [1] for details and the Dirac operator.

The repository also contains `analysis_autocorr.py` that computes the variances of plquettes with different numbers of blocking to show the autocorrelation (you need to install `gvar` to use it).

[1]: <https://arxiv.org/abs/2208.02728>

