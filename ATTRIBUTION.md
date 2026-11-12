# Attribution

pdeView's own code is MIT-licensed (see [`LICENSE`](LICENSE)). This file lists the third-party material it uses.

- The **full app** (branch `main`) downloads the PDEBench data and checkpoint at runtime. It does not
  redistribute them.
- The **static demo** at <https://mshibashish.github.io/pdeView/> (built from branch `pages`) **does**
  redistribute PDEBench material, as described below.

Neither the PDEBench authors nor DaRUS endorse pdeView. Their material is provided as is, without warranties
(CC BY 4.0, Section 5).

## PDEBench 2D shallow-water data (CC BY 4.0)

> Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022).
> *PDEBench Datasets* (V8) [Dataset]. DaRUS. <https://doi.org/10.18419/darus-2986>

- File: `2D/shallow-water/2D_rdb_NA_NA.h5` (radial dam break), MD5 `75d838c47aa410694bdc912ea7f22282`.
- Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

What the static demo contains, and the changes made:

| Content | Changes |
|---|---|
| Ground-truth water height *h* of 110 samples: the test split (0900–0999) and 10 training samples, with the grid and time coordinates | **Unchanged values** (float32, bit-identical to the file). Only the subset and the container differ: each `[101, 128, 128]` field (channel axis dropped) is stored losslessly as `static-api/samples/<key>/gt.bin` (byte-shuffled, gzip). |
| FNO predictions for every sample | **Derived and quantised.** Rolled out in CI with the checkpoint below from the first 10 ground-truth frames, then stored as 16-bit integers (`q = round((h − lo) / scale)`, delta-coded along time). The maximum absolute error is half a quantisation step, about 3e-5 or less in *h*. The exact bound is `pred_max_abs_error` in `static-api/manifest.json`. |
| Error statistics, reports and figures | **Derived.** Computed from the data and the predictions; the statistics use the unquantised fields. |

Not PDEBench data: the ground truth of the 200 held-out cases, which pdeView computed with PyClaw (see below), and
the custom dam breaks that the demo computes in the visitor's browser.

## PDEBench pretrained FNO (CC BY 4.0)

> Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022).
> *PDEBench Pretrained Models* (Version 2.0) [Dataset]. DaRUS. <https://doi.org/10.18419/darus-2987>

- File: `fno/2D_rdb_NA_NA_FNO.pt`, MD5 `e75d87abd5f6a7b777cb2f0e6a3a05c4`.
- Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Changes in the static demo:** the checkpoint is shipped **converted to ONNX** as
  `static-api/model/fno_step.onnx`, one autoregressive step of the network. The network is not retrained or
  fine-tuned; the trained parameters are only re-expressed for ONNX. The FFTs are replaced by the equivalent
  truncated real DFT written as dense matrix products, and the coordinate grid is baked in as a constant. The
  export fails if a 91-step rollout differs from PyTorch by more than 1e-4 (max abs) on a test sample; the
  measured difference is recorded in `static-api/model/model.json`.

## PDEBench code (MIT)

`backend/pdeview/model/fno.py`, from which the ONNX model is exported, is adapted from PDEBench's
[`pdebench/models/fno/fno.py`](https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/fno.py).

- PDEBench is released under the MIT license "except where otherwise stated":
  Copyright 2022 NEC Labs Europe GmbH, Stuttgart University, CSIRO and PDEbench contributors.
- `fno.py` itself is "taken and modified from https://github.com/zongyi-li/fourier_neural_operator", under the
  MIT License, Copyright (c) 2020 Zongyi Li.

pdeView's metrics are its own implementation of PDEBench's metric definitions. No code was taken from PDEBench's
`metrics.py`, which is under a separate non-commercial licence.

## Papers

PDEBench:

> Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022).
> PDEBench: An Extensive Benchmark for Scientific Machine Learning. In *Advances in Neural Information Processing
> Systems 35* (NeurIPS 2022), Datasets and Benchmarks Track, pp. 1596–1611.
> <https://doi.org/10.52202/068431-0117>, [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)

The PDEBench authors ask users of the data or code to cite the paper and the dataset. The BibTeX from the
[PDEBench README](https://github.com/pdebench/PDEBench#publications--citations):

```bibtex
@inproceedings{PDEBench2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
title = {{PDEBench: An Extensive Benchmark for Scientific Machine Learning}},
year = {2022},
booktitle = {36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
url = {https://arxiv.org/abs/2210.07182}
}

@data{darus-2986_2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
publisher = {DaRUS},
title = {{PDEBench Datasets}},
year = {2022},
doi = {10.18419/darus-2986},
url = {https://doi.org/10.18419/darus-2986}
}
```

Fourier Neural Operator:

> Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021).
> Fourier Neural Operator for Parametric Partial Differential Equations. *ICLR 2021*.
> [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

## Frontend libraries bundled into the site

| Library (npm packages) | Licence | Copyright notice | Used by |
|---|---|---|---|
| [React](https://github.com/react/react) (`react`, `react-dom`, `scheduler`) | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. | full app and static demo |
| [three.js](https://github.com/mrdoob/three.js) (`three`, including `OrbitControls`) | MIT | Copyright © 2010-2026 three.js authors | full app and static demo |
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (`onnxruntime-web`) | MIT | Copyright (c) Microsoft Corporation | static demo only |
| [Vite](https://github.com/vitejs/vite) (build tool; adds small runtime helpers to the bundle) | MIT | Copyright (c) 2019-present, VoidZero Inc. and Vite contributors | full app and static demo |

`onnxruntime-web` includes ONNX Runtime compiled to WebAssembly (`ort-wasm-*.wasm`). ONNX Runtime incorporates
third-party components, whose notices are in its
[`ThirdPartyNotices.txt`](https://github.com/microsoft/onnxruntime/blob/main/ThirdPartyNotices.txt). The npm
package has no licence file of its own and refers to the ONNX Runtime repository.

The exact versions are pinned in `frontend/package-lock.json`. All four are under the MIT License:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Used offline, not redistributed

- **Clawpack / PyClaw** 5.14.0: [BSD 3-Clause](https://github.com/clawpack/clawpack/blob/master/LICENSE),
  Copyright (c) 1994-2018, Clawpack Developers. `scripts/pyclaw_reference.py` runs PyClaw with PDEBench's
  configuration to compute the ground truth of the 200 held-out cases. The static demo contains those computed
  fields (float32, unchanged) but no Clawpack code. pdeView's own solver (`backend/pdeview/solver/swe.py`, and
  its JavaScript port in the static demo) is an independent implementation and does not use Clawpack code.
  Clawpack asks to be cited as:
  - Clawpack Development Team (2026). *Clawpack software*, version 5.14.0. <http://www.clawpack.org>
  - Ketcheson, D. I., Mandli, K. T., Ahmadia, A. J., Alghamdi, A., Quezada de Luna, M., Parsani, M.,
    Knepley, M. G., & Emmett, M. (2012). PyClaw: Accessible, Extensible, Scalable Tools for Wave Propagation
    Problems. *SIAM Journal on Scientific Computing*, 34(4), C210–C231. <https://doi.org/10.1137/110856976>
- **PyTorch** (BSD-style licence) computes the FNO predictions and exports the ONNX model.
- **ONNX** (Apache 2.0) and **ONNX Runtime** (MIT), as Python packages, build and check the ONNX model.

The other Python dependencies are listed in `pyproject.toml` and `uv.lock`.
