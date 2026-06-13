"""Execution harness run *inside* the sandbox (Docker or local subprocess).

Usage: ``python harness.py <code_path> <images_out_path>``

Runs the user's source with a headless matplotlib backend, lets stdout/stderr
flow normally, then serializes any open figures as base64 PNGs to a JSON file
(kept out of the stdout/stderr streams so they stay clean).
"""

from __future__ import annotations

import base64
import io
import json
import sys
import traceback


def main() -> int:
    code_path, images_path = sys.argv[1], sys.argv[2]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(code_path, encoding="utf-8") as fh:
        source = fh.read()

    exit_code = 0
    namespace: dict = {"__name__": "__main__"}
    try:
        exec(compile(source, "<user_code>", "exec"), namespace)
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 0
    except BaseException:  # noqa: BLE001 - surface any user error as a traceback
        traceback.print_exc()
        exit_code = 1

    images: list[str] = []
    try:
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=100, bbox_inches="tight")
            images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    except Exception:  # pragma: no cover - figure capture is best-effort
        pass

    with open(images_path, "w", encoding="utf-8") as fh:
        json.dump(images, fh)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
