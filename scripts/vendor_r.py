import os
import shutil


def vendor_r():
    project_root = os.getcwd()
    target_dir = os.path.join(project_root, "package-r", "src", "rust", "core")

    print(f"Vendoring core Rust logic into {target_dir}...")

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


if __name__ == "__main__":
    vendor_r()
