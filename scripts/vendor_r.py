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
    skip_workspace = False
    for line in lines:
        if line.startswith("[workspace]"):
            skip_workspace = True
            continue
        if skip_workspace:
            if line.startswith("[") or line.strip() == "":
                if line.startswith("["):
                    skip_workspace = False
                else:
                    continue
            else:
                continue

        # Stop at [dev-dependencies]
        if line.startswith("[dev-dependencies]"):
            new_lines.append("# Dev-dependencies and benches removed for vendoring\n")
            break

        # Comment out license-file and readme
        line = line.replace('license-file = "LICENSE"', '# license-file = "LICENSE"')
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
