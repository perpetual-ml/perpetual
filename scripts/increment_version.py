import argparse
import re
import subprocess
from pathlib import Path

# python scripts/increment_version.py 1.0.28 --dry-run


def update_file(file_path, pattern, replacement, dry_run=False):
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: File not found: {file_path}")
        return

    content = path.read_text(encoding="utf-8")
    new_content = re.sub(pattern, replacement, content, count=1)

    if content != new_content:
        print(f"Updating {file_path}...")
        if not dry_run:
            path.write_text(new_content, encoding="utf-8")
    else:
        print(f"No changes needed for {file_path}")


def update_cargo_toml(file_path, new_version, dry_run=False):
    # Pattern: version = "X.Y.Z"
    # match start of line or whitespace
    pattern = r'(^|^\[package\]\s*\n|^\[.*\]\s*\n|^\s*)version\s*=\s*"[^"]+"'
    # We want to target the 'version = "..."' under [package] ideally, but simple regex usually works if it's high up.
    # A safer regex finding 'version = "..."' specifically.

    # Actually, for Cargo.toml, it's standard to just have `version = "..."`.
    # Let's use a regex that looks for `version = "..."`
    pattern = r'(version\s*=\s*)"[^"]+"'
    replacement = f'\\1"{new_version}"'
    update_file(file_path, pattern, replacement, dry_run)


def update_pyproject_toml(file_path, new_version, dry_run=False):
    # Standard pyproject pattern: version = "X.Y.Z"
    pattern = r'(version\s*=\s*)"[^"]+"'
    replacement = f'\\1"{new_version}"'
    update_file(file_path, pattern, replacement, dry_run)


def update_r_description(file_path, new_version, dry_run=False):
    # Pattern: Version: X.Y.Z
    pattern = r"(Version:\s*)[^\s]+"
    replacement = f"\\g<1>{new_version}"
    update_file(file_path, pattern, replacement, dry_run)


def update_conf_py(file_path, new_version, dry_run=False):
    # Pattern: release = "X.Y.Z"
    pattern = r'(release\s*=\s*)"[^"]+"'
    replacement = f'\\1"{new_version}"'
    update_file(file_path, pattern, replacement, dry_run)


def update_python_cargo_toml(file_path, new_version, dry_run=False):
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: File not found: {file_path}")
        return

    content = path.read_text(encoding="utf-8")

    # 1. Update package version
    # version = "1.0.3"
    package_ver_pattern = r'(version\s*=\s*)"[^"]+"'
    content = re.sub(package_ver_pattern, f'\\1"{new_version}"', content, count=1)

    # 2. Update dependency version
    # perpetual_rs = {package="perpetual", version = "1.0.3", path = "../" }
    # We look for 'perpetual_rs' ... 'version = "..."'
    dep_pattern = r'(perpetual_rs\s*=.*?version\s*=\s*)"[^"]+"'
    content = re.sub(dep_pattern, f'\\1"{new_version}"', content, count=1)

    print(f"Updating {file_path}...")
    if not dry_run:
        path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Increment version numbers across the repo."
    )
    parser.add_argument("version", help="The new version number (e.g., 1.0.4)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print changes without modifying files"
    )

    args = parser.parse_args()
    new_version = args.version
    dry_run = args.dry_run

    root_dir = Path(__file__).parent.parent

    # Files to update
    cargo_toml = root_dir / "Cargo.toml"
    pyproject_toml = root_dir / "package-python" / "pyproject.toml"
    python_cargo_toml = root_dir / "package-python" / "Cargo.toml"
    r_rust_cargo_toml = root_dir / "package-r" / "src" / "rust" / "Cargo.toml"
    r_description = root_dir / "package-r" / "DESCRIPTION"
    conf_py = root_dir / "package-python" / "docs" / "source" / "conf.py"

    print(f"Setting version to: {new_version}")

    update_cargo_toml(cargo_toml, new_version, dry_run)
    update_pyproject_toml(pyproject_toml, new_version, dry_run)
    update_python_cargo_toml(python_cargo_toml, new_version, dry_run)  # Special handler
    update_cargo_toml(r_rust_cargo_toml, new_version, dry_run)  # Standard structure
    update_r_description(r_description, new_version, dry_run)
    update_conf_py(conf_py, new_version, dry_run)

    # Handle uv.lock
    if not dry_run:
        print("Updating uv.lock...")
        try:
            subprocess.run(
                ["uv", "lock"], cwd=root_dir / "package-python", check=True, shell=True
            )
            print("uv.lock updated successfully.")
        except subprocess.CalledProcessError:
            print(
                "Warning: Failed to update uv.lock. Make sure 'uv' is installed and accessible."
            )
        except FileNotFoundError:
            print("Warning: 'uv' command not found. Skipping lock file update.")


if __name__ == "__main__":
    main()
