import pytest

from pdeview.config import CKPT_DIR, SUBSET_FILE, SWE_FNO_CKPT

CKPT_PATH = CKPT_DIR / SWE_FNO_CKPT.name

needs_ckpt = pytest.mark.skipif(not CKPT_PATH.exists(), reason="checkpoint not downloaded")
needs_data = pytest.mark.skipif(not SUBSET_FILE.exists(), reason="data subset not downloaded")
