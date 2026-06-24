"""Build CHEMKIN Rate Viewer v2.0 as a directory-mode Windows package."""

import os
import shutil
import sys
import time

import PyInstaller.__main__


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(ROOT_DIR, "dist")
BUILD_DIR = os.path.join(ROOT_DIR, "build")
SPEC_PATH = os.path.join(ROOT_DIR, "CHEMKIN_RateViewer.spec")


def remove_directory_if_exists(path, allowed_root):
    """Remove generated output only when it is inside the intended workspace."""
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(allowed_root)
    if os.path.commonpath([resolved_path, resolved_root]) != resolved_root:
        raise ValueError(f"Refusing to remove path outside {resolved_root}: {resolved_path}")
    for attempt in range(10):
        if not os.path.exists(resolved_path):
            return
        try:
            shutil.rmtree(resolved_path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(1)


def configure_conda_runtime_dll_path():
    """Expose Conda DLLs to PyInstaller when invoked by python.exe directly."""
    conda_dll_dir = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
    if os.path.isdir(conda_dll_dir):
        os.environ["PATH"] = conda_dll_dir + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("CONDA_PREFIX", os.path.dirname(sys.executable))
        print(f"Using Conda DLL directory: {conda_dll_dir}")


def main():
    os.chdir(ROOT_DIR)
    configure_conda_runtime_dll_path()

    remove_directory_if_exists(BUILD_DIR, ROOT_DIR)
    package_dir = os.path.join(DIST_DIR, "CHEMKIN_RateViewer")
    remove_directory_if_exists(package_dir, DIST_DIR)

    print("=" * 70)
    print("Building CHEMKIN Rate Viewer v2.0 (onefile mode)")
    print("=" * 70)
    PyInstaller.__main__.run(["--clean", "--noconfirm", SPEC_PATH])

    executable = os.path.join(DIST_DIR, "CHEMKIN_RateViewer.exe")
    if not os.path.exists(executable):
        raise FileNotFoundError(f"Packaged executable not found: {executable}")

    print("\n" + "=" * 70)
    print("Build completed")
    print("=" * 70)
    print(f"Package directory: {package_dir}")
    print(f"Executable: {executable}")
    print("User thermo database: %LOCALAPPDATA%\\CHEMKIN_RateViewer\\therm.dat")


if __name__ == "__main__":
    main()
