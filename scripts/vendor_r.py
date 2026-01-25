import os
import shutil
import subprocess


def vendor_r():
    project_root = os.getcwd()
    target_dir = os.path.join(project_root, "package-r", "src", "rust", "core")
    old_vendor_dir = os.path.join(project_root, "package-r", "src", "rust", "vendor")

    print(f"Vendoring core Rust logic into {target_dir}...")

    # 0. Clean up old vendor directory if it exists
    if os.path.exists(old_vendor_dir):
        print(f"Cleaning up old vendor directory {old_vendor_dir}...")
        shutil.rmtree(old_vendor_dir)

    # 1. Clean up existing core directory
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
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

        # Comment out license-file and readme
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
    vendor_dependencies(project_root)


def vendor_dependencies(project_root):
    rust_dir = os.path.join(project_root, "package-r", "src", "rust")
    vendor_dir = os.path.join(project_root, "package-r", "v")
    config_dir = os.path.join(rust_dir, ".cargo")
    config_file = os.path.join(config_dir, "config.toml")

    print(f"Vendoring dependencies into {vendor_dir}...")

    # Run cargo vendor
    # We must run this in the directory containing the Cargo.toml (package-r/src/rust)
    try:
        result = subprocess.run(
            ["cargo", "vendor", "../../v"],
            cwd=rust_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running cargo vendor: {e.stderr}")
        raise e

    # Create .cargo/config.toml with the output from cargo vendor
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # The output from cargo vendor is relative to the cwd where it was run
    # Since we run it in rust_dir (package-r/src/rust) and vendor into "../../v" (package-r/v)
    # The config should point to "../../v"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    print(f"Created {config_file}")

    # 6. Prune unnecessary large/deep files to stay under 100 char path limit
    prune_vendor(vendor_dir)

    # 7. Fix checksums for potential missing files
    fix_checksums(vendor_dir)


def prune_vendor(vendor_dir):
    print(f"Pruning unnecessary files in {vendor_dir}...")
    # Patterns of directories to remove
    to_remove_dirs = [
        "tests",
        "examples",
        "benches",
        "doc",
    ]
    # More aggressive for windows crates which have insane depth
    deep_prune_dirs = [
        "src/Windows/Win32/UI",
        "src/Windows/Win32/Management",
        "src/Windows/Win32/NetworkManagement",
        "src/Windows/Win32/Devices",
        "src/Windows/Win32/System",
        "src/Windows/Win32/Graphics",
        "src/Windows/Win32/Security",
        "src/Windows/Win32/Media",
        "src/Windows/Win32/Storage",
        "src/Windows/Wdk",
    ]

    for root, dirs, files in os.walk(vendor_dir, topdown=False):
        for name in dirs:
            if name in to_remove_dirs:
                shutil.rmtree(os.path.join(root, name))

        # Additional deep prune for windows crates if we are inside one
        if "windows" in root:
            for deep_dir in deep_prune_dirs:
                target = os.path.join(root, deep_dir)
                if os.path.exists(target):
                    shutil.rmtree(target)

    # Remove ALL markdown files as they are often long and redundant
    for root, dirs, files in os.walk(vendor_dir):
        for name in files:
            if name.lower().endswith(".md"):
                os.remove(os.path.join(root, name))


def fix_checksums(vendor_dir):
    print(f"Scanning {vendor_dir} for checksum issues...")
    import json

    if not os.path.exists(vendor_dir):
        print(f"Error: {vendor_dir} does not exist.")
        return

    patched_count = 0

    for root, dirs, files in os.walk(vendor_dir):
        if ".cargo-checksum.json" in files:
            checksum_path = os.path.join(root, ".cargo-checksum.json")
            with open(checksum_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            files_dict = data.get("files", {})
            files_to_remove = []

            for file_rel_path in files_dict.keys():
                # files_dict keys are relative to the crate root (which is 'root')
                full_path = os.path.join(root, file_rel_path)
                if not os.path.exists(full_path):
                    # print(f"Missing file: {file_rel_path} in {root} -> Removing from checksums.")
                    files_to_remove.append(file_rel_path)

            if files_to_remove:
                for fpath in files_to_remove:
                    del files_dict[fpath]

                # Write back
                with open(checksum_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=None, separators=(",", ":"))

                patched_count += 1
                # print(f"Patched {checksum_path}")

    print(f"Done. Patched {patched_count} crates.")


if __name__ == "__main__":
    vendor_r()
