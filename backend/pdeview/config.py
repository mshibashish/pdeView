"""Central configuration: paths, remote asset ids, checksums and reference numbers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PDEVIEW_DATA_DIR", ROOT / "data"))
CKPT_DIR = Path(os.environ.get("PDEVIEW_CKPT_DIR", ROOT / "checkpoints"))
REPORTS_DIR = Path(os.environ.get("PDEVIEW_REPORTS_DIR", ROOT / "reports"))
FRONTEND_DIST = Path(os.environ.get("PDEVIEW_FRONTEND_DIST", ROOT / "frontend" / "dist"))

DARUS_ACCESS = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    file_id: int
    size: int
    md5: str

    @property
    def url(self) -> str:
        return DARUS_ACCESS.format(file_id=self.file_id)


# PDEBench datasets, doi:10.18419/darus-2986 (CC BY 4.0)
SWE_DATA = RemoteFile(
    name="2D_rdb_NA_NA.h5",
    file_id=133021,
    size=6_626_098_972,
    md5="75d838c47aa410694bdc912ea7f22282",
)

# PDEBench pretrained models, doi:10.18419/darus-2987 (CC BY 4.0)
SWE_FNO_CKPT = RemoteFile(
    name="2D_rdb_NA_NA_FNO.pt",
    file_id=133066,
    size=11_135_711,
    md5="e75d87abd5f6a7b777cb2f0e6a3a05c4",
)

SUBSET_FILE = DATA_DIR / "swe_subset.h5"

# PDEBench FNO hyper-parameters for 2D_rdb_NA_NA (pdebench/models/config/args/config_rdb.yaml)
FNO_NUM_CHANNELS = 1
FNO_MODES = 12
FNO_WIDTH = 20
FNO_INITIAL_STEP = 10

# Split used by PDEBench's default loader (FNODatasetMult, test_ratio=0.1): the last 10 % of
# the sorted sample keys are held out for testing.
N_SAMPLES_TOTAL = 1000
TEST_RATIO = 0.1
N_TRAIN_PROBE = 10  # a few training samples, used to look for over-fitting / split leakage

# Reported FNO errors for the shallow-water equation (PDEBench paper, arXiv:2210.07182v7, Table 5)
PAPER_FNO_SWE = {
    "RMSE": 4.5e-3,
    "nRMSE": 4.4e-3,
    "max_error": 4.5e-2,
    "cRMSE": 2.0e-4,
    "bRMSE": 1.4e-3,
    "fRMSE_low": 2.6e-4,
    "fRMSE_mid": 3.1e-4,
    "fRMSE_high": 2.5e-4,
}
