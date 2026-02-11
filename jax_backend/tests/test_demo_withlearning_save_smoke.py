import os
import sys
from pathlib import Path

import numpy as np


os.environ.setdefault("MPLBACKEND", "Agg")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import demo_withlearning


def test_demo_withlearning_save_smoke(tmp_path: Path) -> None:
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        demo_withlearning.run(
            init_key_num=7,
            N=6,
            T=1,
            dt=0.02,
            average_sz=2.0,
            save=True,
        )
    finally:
        os.chdir(original_cwd)

    output_file = tmp_path / "sim_hist_key7.npz"
    assert output_file.exists()

    data = np.load(output_file)
    assert data["r"].ndim == 3
    assert data["v"].ndim == 3
    assert data["r"].shape[-1] == 2
    assert data["v"].shape[-1] == 2
