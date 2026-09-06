"""Ensure a third-party scripts package cannot shadow project entry points."""
import json
from pathlib import Path
import subprocess
import sys


def test_project_scripts_precedes_installed_package(tmp_path):
    root = Path(__file__).resolve().parents[1]
    foreign = tmp_path / 'scripts'
    foreign.mkdir()
    (foreign / '__init__.py').write_text('raise RuntimeError("wrong scripts package")')
    code = (
        'import sys, pathlib; '
        f'sys.path.insert(0, {str(tmp_path)!r}); '
        f'sys.path.insert(0, {str(root)!r}); '
        'import scripts; '
        f'assert pathlib.Path(scripts.__file__).resolve() == pathlib.Path({str(root / "scripts" / "__init__.py")!r}).resolve(); '
        'from scripts.train_skeleton_seq2seq import SkeletonSeq2SeqModel; '
        'from scripts.run_fixed_pipeline import main'
    )
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
