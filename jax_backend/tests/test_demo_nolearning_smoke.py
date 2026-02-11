import os
import sys
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import demo_nolearning


def test_demo_nolearning_tiny_run_defaults(tmp_path: Path) -> None:
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        demo_nolearning.run(
            init_key_num=1,
            N=4,
            T=0.2,
            dt=0.02,
            last_T_seconds=0.1,
            save=True,
        )
    finally:
        os.chdir(original_cwd)

    assert (tmp_path / "sim_hist_key1.npz").exists()
