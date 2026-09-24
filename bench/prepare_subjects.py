"""Fetch the pinned benchmark subjects into bench/subjects/<name>/ (gitignored).

For each subject: download the wheel from PyPI (no dependencies), keep only the import
package (dropping any bundled test packages) and the license files, and write a
requirements.txt with the subject's runtime dependencies for the sandbox image.

    python bench/prepare_subjects.py            # all subjects
    python bench/prepare_subjects.py slugify    # some
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS_FILE = ROOT / "bench" / "subjects.yaml"
OUT = ROOT / "bench" / "subjects"
TEST_DIRS = {"test", "tests", "testing"}


def load_subjects(names: list[str] | None = None) -> list[dict]:
    subjects = yaml.safe_load(SUBJECTS_FILE.read_text(encoding="utf-8"))["subjects"]
    return [s for s in subjects if not names or s["name"] in names]


def prepare(subject: dict) -> Path:
    dest = OUT / subject["name"]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary",
                        ":all:", "-q", "-d", tmp, subject["package"]], check=True)
        wheel = next(Path(tmp).glob("*.whl"))
        with zipfile.ZipFile(wheel) as zf:
            for info in zf.infolist():
                parts = Path(info.filename).parts
                if parts[0] == subject["import_root"] and not TEST_DIRS & set(parts[1:-1]):
                    zf.extract(info, dest)
                elif parts[0].endswith(".dist-info") and "LICENSE" in parts[-1].upper():
                    (dest / "LICENSE").write_bytes(zf.read(info))
    requirements = subject.get("requirements", [])
    (dest / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    (dest / "SOURCE.txt").write_text(
        f"{subject['package']} (wheel from PyPI); tests removed. See LICENSE.\n",
        encoding="utf-8")
    return dest


def describe(subject: dict) -> str:
    target = OUT / subject["name"] / subject["target"]
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
    classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
    return (f"{subject['name']:<24} {subject['level']:<7} {len(source.splitlines()):>5} lines, "
            f"{functions:>3} functions, {classes:>2} classes  ({subject['target']})")


def main() -> None:
    for subject in load_subjects(sys.argv[1:] or None):
        prepare(subject)
        print(describe(subject))


if __name__ == "__main__":
    main()
