from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from common import PROJECT_ROOT, qresafe_root_from_arg, qresafe_commit  # noqa: E402


def main(output: Path, qresafe_root_arg: str | None):
    lines = [
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"executable={sys.executable}",
        f"qresafe_quant_without_ft_commit={qresafe_commit(qresafe_root_from_arg(qresafe_root_arg))}",
    ]
    try:
        import torch

        lines.extend([
            f"torch={torch.__version__}",
            f"torch_cuda_build={torch.version.cuda}",
            f"cuda_available={torch.cuda.is_available()}",
            f"gpu_count={torch.cuda.device_count()}",
        ])
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                lines.append(f"gpu_{index}={torch.cuda.get_device_name(index)}")
    except ImportError:
        lines.append("torch=not-installed")
    try:
        import transformers

        lines.append(f"transformers={transformers.__version__}")
    except ImportError:
        lines.append("transformers=not-installed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "environment.txt")
    parser.add_argument("--qresafe-root", default=None)
    args = parser.parse_args()
    main(args.output, args.qresafe_root)
