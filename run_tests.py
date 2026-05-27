#!/usr/bin/env python
"""
Single-entry test runner.

Runs the full pytest suite with coverage and prints the report. Exits non-zero
if any test fails or coverage drops below 100%.

Usage:
    py run_tests.py              # run everything
    py run_tests.py tests/test_safety.py   # forward args to pytest
    py run_tests.py -k pipeline  # filter
"""

import subprocess
import sys


def main() -> int:
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov",
        "--cov-report=term-missing",
        "--cov-fail-under=100",
    ]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    else:
        cmd.append("tests/")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
