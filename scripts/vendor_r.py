import os
import re
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

    # 3. Copy and transform Cargo.toml
    cargo_path = os.path.join(project_root, "Cargo.toml")
    target_cargo_path = os.path.join(target_dir, "Cargo.toml")

    with open(cargo_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove [workspace] section
    content = re.sub(r"(?s)\[workspace\].*?members = \[.*?\]", "", content)

    # Remove [dev-dependencies] and everything after it
    content = re.sub(
        r"(?s)\[dev-dependencies\].*",
        "# Dev-dependencies and benches removed for vendoring\n",
        content,
    )

    # Comment out license-file and readme
    content = content.replace('license-file = "LICENSE"', '# license-file = "LICENSE"')
    content = content.replace('readme = "README.md"', '# readme = "README.md"')

    # Clean up multiple newlines
    content = re.sub(r"\n{3,}", "\n\n", content)

    with open(target_cargo_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Vendoring complete.")


if __name__ == "__main__":
    vendor_r()
