"""Build, inspect, install and upload this project's current release."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT, env=None):
    subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True)


def version():
    namespace = {}
    exec((ROOT / "was_disaggregation/_version.py").read_text(), namespace)
    return namespace["__version__"]


def artifacts():
    stem = "was_disaggregation-" + version()
    expected = [ROOT / "dist" / (stem + "-py3-none-any.whl"),
                ROOT / "dist" / (stem + ".tar.gz")]
    if not all(p.is_file() for p in expected):
        raise RuntimeError("Build the current wheel and sdist first: pixi run build")
    return expected


def build():
    # Remove only this project's old distributions; uploads use the exact pair.
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    for pattern in ("was_disaggregation-*.whl", "was_disaggregation-*.tar.gz"):
        for path in dist.glob(pattern):
            path.unlink()
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    run(sys.executable, "-m", "build", "--no-isolation")


def check():
    files = artifacts()
    run(sys.executable, "-m", "twine", "check", "--strict", *files)
    with zipfile.ZipFile(files[0]) as wheel:
        names = wheel.namelist()
        for source in (ROOT / "was_disaggregation").glob("*.py"):
            if "was_disaggregation/" + source.name not in names:
                raise RuntimeError(f"Wheel omits {source.name}")
        if not any(name.endswith("/licenses/LICENSE") for name in names):
            raise RuntimeError("Wheel omits GPLv3 license")


def verify_wheel():
    wheel = artifacts()[0]
    with tempfile.TemporaryDirectory(prefix="was-wheel-") as tmp:
        directory = Path(tmp)
        # Reuse scientific dependencies, while installing the wheel in a clean
        # venv and running outside the source tree. Do not resolve from TestPyPI.
        venv.EnvBuilder(with_pip=True).create(directory / "env")
        executable = directory / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = os.environ.copy()
        # A nested venv does not inherit the parent venv's scientific packages.
        # Append the invoking interpreter's dependency paths *after* the new
        # environment's site-packages so the installed wheel always wins.
        site_dir = Path(subprocess.check_output(
            [str(executable), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            cwd=directory, text=True).strip())
        dependency_paths = [str(path) for entry in sys.path if entry
                            for path in [Path(entry).resolve()]
                            # Pixi and local venv dependencies live below ROOT.
                            # Exclude only the source import directory, not its
                            # entire subtree, or an actual Pixi wheel check loses
                            # NumPy and all other parent-environment packages.
                            if path.is_dir() and path != ROOT.resolve()
                            and not path.is_relative_to(directory)]
        (site_dir / "_was_parent_dependencies.pth").write_text(
            "\n".join(dict.fromkeys(dependency_paths)) + "\n", encoding="utf-8")
        env.pop("PYTHONPATH", None)
        run(executable, "-m", "pip", "install", "--no-deps", wheel, cwd=directory, env=env)
        code = (
            "from pathlib import Path; import was_disaggregation as w; "
            "from importlib.metadata import version; "
            f"assert Path(w.__file__).is_relative_to(Path({str(directory)!r})); "
            "assert w.__version__ == version('was-disaggregation'); "
            "assert all(hasattr(w, name) for name in w.__all__); "
            "print('Installed wheel:', w.__version__, w.__file__)"
        )
        run(executable, "-c", code, cwd=directory, env=env)
        command = directory / "env" / ("Scripts/was-disaggregation.exe" if os.name == "nt" else "bin/was-disaggregation")
        run(command, "--version", cwd=directory, env=env)
        run(command, "--help", cwd=directory, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "verify-wheel", "upload"))
    parser.add_argument("--repository", choices=("testpypi", "pypi"))
    args = parser.parse_args()
    if args.action == "upload":
        if not args.repository:
            parser.error("upload requires --repository")
        check()
        # Twine prompts securely for the API token if no keyring/env token exists.
        run(sys.executable, "-m", "twine", "upload", "--repository", args.repository,
            "--username", "__token__", *artifacts())
    else:
        {"build": build, "check": check, "verify-wheel": verify_wheel}[args.action]()


if __name__ == "__main__":
    main()
