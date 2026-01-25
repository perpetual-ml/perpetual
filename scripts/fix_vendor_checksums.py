import argparse
import json
import os


def fix_checksums(vendor_dir):
    print(f"Scanning {vendor_dir} for checksum issues...")

    if not os.path.exists(vendor_dir):
        print(f"Error: {vendor_dir} does not exist.")
        return

    patched_count = 0

    for root, dirs, files in os.walk(vendor_dir):
        if ".cargo-checksum.json" in files:
            checksum_path = os.path.join(root, ".cargo-checksum.json")
            with open(checksum_path, "r") as f:
                data = json.load(f)

            files_dict = data.get("files", {})
            files_to_remove = []

            for file_rel_path in files_dict.keys():
                # files_dict keys are relative to the crate root (which is 'root')
                full_path = os.path.join(root, file_rel_path)
                if not os.path.exists(full_path):
                    print(
                        f"Missing file: {file_rel_path} in {root} -> Removing from checksums."
                    )
                    files_to_remove.append(file_rel_path)

            if files_to_remove:
                for fpath in files_to_remove:
                    del files_dict[fpath]

                # Write back
                with open(checksum_path, "w") as f:
                    json.dump(data, f, indent=None, separators=(",", ":"))

                patched_count += 1
                print(f"Patched {checksum_path}")

    print(f"Done. Patched {patched_count} crates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-dir",
        default="package-r/src/rust/vendor",
        help="Path to vendor directory",
    )
    args = parser.parse_args()

    fix_checksums(args.vendor_dir)
