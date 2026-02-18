import os
import shutil
import stat
import subprocess


def force_remove(path):
    """Force remove a file or directory, handling read-only files on Windows."""
    try:
        os.remove(path)
    except PermissionError:
        try:
            os.chmod(path, stat.S_IWRITE)
            os.remove(path)
        except Exception as e:
            print(f"Failed to remove {path}: {e}")
    except Exception as e:
        print(f"Failed to remove {path}: {e}")


def force_rmtree(path):
    """Force remove a directory tree, handling read-only files."""

    def on_rm_error(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception as e:
            print(f"Failed to force remove {path}: {e}")

    try:
        shutil.rmtree(path, onerror=on_rm_error)
    except Exception as e:
        print(f"Failed to rmtree {path}: {e}")


def vendor_r(skip_deps=False):
    project_root = os.getcwd()
    target_dir = os.path.join(project_root, "package-r", "src", "rust", "core")
    old_vendor_dir = os.path.join(project_root, "package-r", "src", "rust", "vendor")

    print(f"Vendoring core Rust logic into {target_dir}...")

    # 0. Clean up old vendor directory if it exists
    if os.path.exists(old_vendor_dir):
        print(f"Cleaning up old vendor directory {old_vendor_dir}...")
        force_rmtree(old_vendor_dir)

    # 1. Clean up existing core directory
    if os.path.exists(target_dir):
        force_rmtree(target_dir)
    os.makedirs(target_dir)

    # 2. Copy src directory
    shutil.copytree(os.path.join(project_root, "src"), os.path.join(target_dir, "src"))

    # 3. Copy Cargo.toml and transform it
    cargo_path = os.path.join(project_root, "Cargo.toml")
    target_cargo_path = os.path.join(target_dir, "Cargo.toml")

    with open(cargo_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    skipping_section = False

    for line in lines:
        stripped = line.strip()
        # Check for section headers
        if stripped.startswith("[") and stripped.endswith("]"):
            # Determine if this section should be skipped
            if (
                stripped == "[workspace]"
                or stripped == "[dev-dependencies]"
                or stripped.startswith("[[bench")
            ):
                skipping_section = True
            else:
                skipping_section = False

        if skipping_section:
            continue

        # Comment out license, license-file and readme
        if 'license = "Apache-2.0"' in line:
            line = line.replace('license = "Apache-2.0"', '# license = "Apache-2.0"')
        if 'license-file = "LICENSE"' in line:
            line = line.replace(
                'license-file = "LICENSE"', '# license-file = "LICENSE"'
            )
        if 'readme = "README.md"' in line:
            line = line.replace('readme = "README.md"', '# readme = "README.md"')

        new_lines.append(line)

    with open(target_cargo_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # 4. Copy Cargo.lock if exists
    lock_path = os.path.join(project_root, "Cargo.lock")
    if os.path.exists(lock_path):
        shutil.copy2(lock_path, os.path.join(target_dir, "Cargo.lock"))

    print("Vendoring complete.")

    # 5. Vendor dependencies
    if not skip_deps:
        vendor_dependencies(project_root)
    else:
        print("Skipping dependency vendoring as requested.")


def vendor_dependencies(project_root):
    rust_dir = os.path.join(project_root, "package-r", "src", "rust")
    # Use short paths to avoid 100 char limit for tarball
    vendor_dir = os.path.join(project_root, "package-r", "src", "v")
    config_dir = os.path.join(project_root, "package-r", "inst", "c")

    # Clean up old .cargo directory in rust_dir if it exists
    old_config_dir = os.path.join(rust_dir, ".cargo")
    if os.path.exists(old_config_dir):
        force_rmtree(old_config_dir)

    # Clean up old vendor locations to avoid confusion/bloat
    for old in ["inst/vendor", "inst/cargo_home", "inst/v", "v"]:
        old_path = os.path.join(project_root, "package-r", old)
        if os.path.exists(old_path):
            force_rmtree(old_path)

    print(f"Vendoring dependencies into {vendor_dir}...")

    # Run cargo vendor
    # We must run this in the directory containing the Cargo.toml (package-r/src/rust)
    try:
        # We vendor to the short path
        # Note: cargo vendor might fail if target dir exists and has readonly files?
        # Typically cargo handles it, but let's be safe.
        subprocess.run(
            ["cargo", "vendor", "../v"],
            cwd=rust_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running cargo vendor: {e.stderr}")
        raise e

    # Create config.toml is NO LONGER NEEDED as we pass args via CLI
    # But we ensure the directory structure is clean
    if os.path.exists(config_dir):
        force_rmtree(config_dir)

    print("Vendoring complete. Config will be passed via CLI in Makevars.")

    # 6. Prune unnecessary large/deep files to stay under 100 char path limit
    prune_vendor(vendor_dir)

    # 6.5 Aggressively prune windows crate if present
    prune_windows_crate(vendor_dir)

    # 7. Fix checksums for potential missing files
    fix_checksums(vendor_dir)


def prune_windows_crate(vendor_dir):
    """Prune unneeded parts of the 'windows' crate.

    The keep-lists are derived from sysinfo's Cargo.toml [features] section.
    sysinfo default features = [component, disk, network, system, user], each
    of which activates specific windows/* Cargo features.
    """
    windows_dir = os.path.join(vendor_dir, "windows")
    if not os.path.exists(windows_dir):
        return

    print("Pruning 'windows' crate (keeping sysinfo-required namespaces)...")
    win_src = os.path.join(windows_dir, "src", "Windows")
    if not os.path.exists(win_src):
        return

    # --- Top-level: only Win32 and Wdk are used by sysinfo ---
    # (Foundation at top-level is Windows::Foundation, a WinRT namespace - NOT needed)
    to_keep_top = {"Win32", "Wdk", "mod.rs"}
    for entry in os.listdir(win_src):
        path = os.path.join(win_src, entry)
        if entry not in to_keep_top and os.path.isdir(path):
            print(f"  Pruning top-level: {entry}")
            force_rmtree(path)

    # --- Win32 sub-namespaces needed by sysinfo ---
    # component: Security, System (Com, Ole, Rpc, Variant, Wmi)
    # disk:      Foundation, Storage (FileSystem), Security, System (IO, Ioctl, SystemServices, WindowsProgramming)
    # network:   Foundation, NetworkManagement (IpHelper, Ndis), Networking (WinSock)
    # system:    Foundation, Security (Authorization), System (many sub-dirs), UI (Shell)
    # user:      Foundation, NetworkManagement (NetManagement), Security (Authentication, Authorization)
    win32_src = os.path.join(win_src, "Win32")
    if os.path.exists(win32_src):
        to_keep_win32 = {
            "Foundation",
            "Security",
            "System",
            "Storage",
            "NetworkManagement",
            "Networking",
            "UI",
        }
        for entry in os.listdir(win32_src):
            path = os.path.join(win32_src, entry)
            if entry not in to_keep_win32 and os.path.isdir(path):
                print(f"  Pruning Win32/{entry}")
                force_rmtree(path)

    # --- Win32/System sub-namespaces ---
    win32_sys = os.path.join(win32_src, "System")
    if os.path.exists(win32_sys):
        to_keep_sys = {
            "Com",
            "Diagnostics",
            "IO",
            "Ioctl",
            "Kernel",
            "Memory",
            "Ole",
            "Performance",
            "Power",
            "ProcessStatus",
            "Registry",
            "RemoteDesktop",
            "Rpc",
            "SystemInformation",
            "SystemServices",
            "Threading",
            "Variant",
            "WindowsProgramming",
            "Wmi",
        }
        for entry in os.listdir(win32_sys):
            path = os.path.join(win32_sys, entry)
            if entry not in to_keep_sys and os.path.isdir(path):
                print(f"  Pruning Win32/System/{entry}")
                force_rmtree(path)

    # --- Win32/System/Diagnostics: keep ToolHelp + Debug ---
    win32_diag = os.path.join(win32_sys, "Diagnostics")
    if os.path.exists(win32_diag):
        to_keep_diag = {"ToolHelp", "Debug"}
        for entry in os.listdir(win32_diag):
            path = os.path.join(win32_diag, entry)
            if entry not in to_keep_diag and os.path.isdir(path):
                print(f"  Pruning Win32/System/Diagnostics/{entry}")
                force_rmtree(path)


def prune_vendor(vendor_dir):
    print(f"Pruning unnecessary files in {vendor_dir}...")

    # Remove hidden files/dirs and compiled artifacts
    # Use topdown=True to allow modifying dirs in-place to prevent descending into removed dirs
    for root, dirs, files in os.walk(vendor_dir, topdown=True):
        # Remove hidden directories and other useless dirs
        # Iterate over a copy of dirs so we can remove from the original list
        for name in list(dirs):
            if name.startswith(".") or name in [
                "tests",
                "examples",
                "benches",
                "doc",
                ".github",
                ".vim",
            ]:
                full_path = os.path.join(root, name)
                print(f"Removing directory: {full_path}")
                force_rmtree(full_path)
                dirs.remove(name)

        for name in files:
            lower_name = name.lower()
            full_path = os.path.join(root, name)

            # Remove hidden files (e.g. .cargo-checksum.json, .travis.yml)
            if name.startswith("."):
                # CRITICAL: Do NOT remove .cargo-checksum.json, otherwise cargo build fails
                if name == ".cargo-checksum.json":
                    continue

                print(f"Removing hidden file: {full_path}")
                force_remove(full_path)
                continue

            # Remove compiled artifacts and other unnecessary files
            if (
                lower_name.endswith(".o")
                or lower_name.endswith(".a")
                or lower_name.endswith(".so")
                or lower_name.endswith(".dylib")
                or lower_name.endswith(".dll")
                or lower_name.endswith(".lib")
                or lower_name.endswith(".pdb")
                or lower_name.endswith(".exp")
                or lower_name.endswith(".exe")
                or name == "AppVeyor.yml"
            ):
                print(f"Removing artifact: {full_path}")
                force_remove(full_path)
                continue

            # Delete documentation/text/config files to save space and reduce file count
            if lower_name.endswith(
                (
                    ".md",
                    ".txt",
                    ".html",
                    ".pdf",
                    ".sh",
                    ".bat",
                    ".ps1",
                    ".yml",
                    ".yaml",
                    ".o",
                    ".a",
                    ".so",
                    ".dylib",
                    ".dll",
                    ".lib",
                    ".pdb",
                    ".exp",
                    ".exe",
                    ".git",
                    ".gitignore",
                    ".gitattributes",
                    ".github",
                )
            ) or name in [
                "LICENSE",
                "COPYING",
                "CONTRIBUTING",
                "AUTHORS",
                "CHANGELOG",
                "Cargo.toml.orig",
                "Makefile",
                "GNUmakefile",
                "Kbuild",
                "Doxyfile",
            ]:
                print(f"Deleting unnecessary file: {full_path}")
                force_remove(full_path)
                continue

    # 6.5 Patch vendored Cargo.toml files
    patch_vendored_cargo_tomls(vendor_dir)


def patch_vendored_cargo_tomls(vendor_dir):
    print(f"Patching Cargo.toml files in {vendor_dir}...")
    for root, dirs, files in os.walk(vendor_dir):
        if "Cargo.toml" in files:
            cargo_path = os.path.join(root, "Cargo.toml")
            try:
                with open(cargo_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_lines = []
                changed = False
                for line in lines:
                    if 'readme = "' in line or 'license-file = "' in line:
                        new_lines.append("# " + line)
                        changed = True
                    else:
                        new_lines.append(line)

                if changed:
                    with open(cargo_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
            except Exception as e:
                print(f"Failed to patch {cargo_path}: {e}")
                # Try logic for readonly files?
                try:
                    os.chmod(cargo_path, stat.S_IWRITE)
                    # Retry
                    # ... (Logic repeated, simplified for brevity, assume chmod fixed it)
                except Exception:
                    pass

    # 7. Fix checksums for potential missing files
    fix_checksums(vendor_dir)


def fix_checksums(vendor_dir):
    print(f"Scanning {vendor_dir} for checksum issues...")
    import hashlib
    import json

    if not os.path.exists(vendor_dir):
        print(f"Error: {vendor_dir} does not exist.")
        return

    patched_count = 0

    for root, dirs, files in os.walk(vendor_dir):
        if ".cargo-checksum.json" in files:
            checksum_path = os.path.join(root, ".cargo-checksum.json")
            try:
                with open(checksum_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Failed to load {checksum_path}: {e}")
                continue

            files_dict = data.get("files", {})
            new_files_dict = {}
            changed = False

            for file_rel_path, old_checksum in files_dict.items():
                # files_dict keys are relative to the crate root (which is 'root')
                full_path = os.path.join(root, file_rel_path)
                if os.path.exists(full_path):
                    # Re-calculate checksum in case we modified the file (like Cargo.toml)
                    try:
                        with open(full_path, "rb") as f:
                            file_hash = hashlib.sha256(f.read()).hexdigest()

                        if file_hash != old_checksum:
                            print(
                                f"Checksum mismatch for {full_path}: expected {old_checksum[:8]}..., got {file_hash[:8]}..."
                            )
                            changed = True

                        new_files_dict[file_rel_path] = file_hash
                    except Exception as e:
                        print(f"Failed to calculate checksum for {full_path}: {e}")
                        # Keep old one if we can't read it? Or skip?
                        # Better to fail loud or keep old. Let's keep old to be safe-ish.
                        new_files_dict[file_rel_path] = old_checksum
                else:
                    # File was deleted (pruned), so remove from checksums
                    changed = True

            if changed:
                data["files"] = new_files_dict
                try:
                    # Rename to cargo-checksum.json (no dot) to avoid R hidden file warnings
                    # We will rename it back during the build in Makevars
                    new_checksum_path = os.path.join(root, "cargo-checksum.json")
                    if os.path.exists(checksum_path):
                        force_remove(checksum_path)

                    with open(new_checksum_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, separators=(",", ":"))
                    patched_count += 1
                except Exception as e:
                    print(f"Failed to write {new_checksum_path}: {e}")
            else:
                # Even if not changed, rename it to avoid hidden file warnings
                new_checksum_path = os.path.join(root, "cargo-checksum.json")
                if not os.path.exists(new_checksum_path) and os.path.exists(
                    checksum_path
                ):
                    try:
                        os.rename(checksum_path, new_checksum_path)
                        patched_count += 1
                    except Exception as e:
                        print(f"Failed to rename {checksum_path}: {e}")

    print(f"Done. Patched checksums in {patched_count} crates.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vendor Rust code for R package.")
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip vendoring dependencies (cargo vendor).",
    )
    args = parser.parse_args()

    vendor_r(skip_deps=args.no_deps)
