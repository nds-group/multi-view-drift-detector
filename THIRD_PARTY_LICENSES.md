# Third-party licenses

This repository is released under the MIT License (see `LICENSE`), except for
the files listed below, which are adapted from third-party code and remain
under their original license.

## river

`ADWIN_CLASS.py` and `ADWIN_IMPLEMENTATION.py` are adapted from the
[river](https://github.com/online-ml/river) library (ADWIN drift detector and
its base classes), with modifications by the KALEIDO authors. river is
distributed under the BSD 3-Clause License:

```
BSD 3-Clause License

Copyright (c) 2020, the river developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## uncertainty-drift-detection

`uncertainty-drift-detection/01_Artificial_DataSets/Data_Friedman_RegressionData_Abrupt.ipynb`
is adapted, with modifications by the KALEIDO authors, from the code
accompanying:

> L. Baier, T. Schlör, J. Schöffer, N. Kühl, "Detecting Concept Drift With
> Neural Network Model Uncertainty," arXiv:2107.01873.

Source: https://github.com/anonymous-account-research/uncertainty-drift-detection

The Friedman stream in
`uncertainty-drift-detection/01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv`
was generated with this notebook. The original repository does not include a
license file; this notebook is not covered by the MIT License of this
repository, and all rights in the original code remain with its authors.
