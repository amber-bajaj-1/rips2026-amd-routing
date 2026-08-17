#!/usr/bin/env python3
"""Reject C++ line comments that accidentally continue through a backslash."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = (
    "routing",
    "bellman_ford",
    "delta_stepping",
    "pre-process",
    "tests",
)
SOURCE_SUFFIXES = {".cc", ".cpp", ".cxx", ".cu", ".h", ".hip", ".hpp"}
CONTINUED_LINE_COMMENT = re.compile(r"//.*\\\s*$")


def main() -> None:
  violations: list[str] = []
  for directory in SOURCE_DIRECTORIES:
    for path in (ROOT / directory).rglob("*"):
      if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
        continue
      for line_number, line in enumerate(
          path.read_text(encoding="utf-8").splitlines(), start=1
      ):
        if CONTINUED_LINE_COMMENT.search(line):
          violations.append(f"{path.relative_to(ROOT)}:{line_number}")

  if violations:
    locations = "\n".join(violations)
    raise AssertionError(
        "C++ // comments must not end in a backslash; use a block comment:\n"
        f"{locations}"
    )
  print("C++ comment-continuation source test passed")


if __name__ == "__main__":
  main()
