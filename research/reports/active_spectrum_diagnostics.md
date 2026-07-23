# Active-spectrum diagnostics

This is a pre-estimator diagnostic only. It estimates leading eigenspace concentration from manual ReLU reverse-mode input gradients on the frozen development partition.

| MLP id | grad cols | E4 | E8 | E16 | top eigenvalues |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 16 | 0.545 | 0.798 | 1.000 | 0.133, 0.0819, 0.076, 0.0597, 0.0501 |
| 1 | 16 | 0.612 | 0.843 | 1.000 | 0.284, 0.182, 0.119, 0.112, 0.0831 |
| 2 | 16 | 0.665 | 0.865 | 1.000 | 0.438, 0.192, 0.149, 0.0927, 0.0785 |
