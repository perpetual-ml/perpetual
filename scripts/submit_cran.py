import os
import re
import sys

import requests


def parse_description(package_dir):
    """Parses DESCRIPTION file for package name, version, and maintainer."""
    desc_path = os.path.join(package_dir, "DESCRIPTION")
    with open(desc_path, "r", encoding="utf-8") as f:
        content = f.read()

    pkg = re.search(r"^Package:\s*(.+)", content, re.MULTILINE).group(1)
    version = re.search(r"^Version:\s*(.+)", content, re.MULTILINE).group(1)

    maintainer_match = re.search(
        r"Maintainer:\s*([^<]+)<([^>]+)>", content, re.MULTILINE
    )
    if not maintainer_match:
        # Try Authors@R
        authors_match = re.search(
            r"Authors@R:\s*(.+)", content, re.MULTILINE | re.DOTALL
        )
        if authors_match:
            # Very basic parsing of Authors@R to find maintainer
            # Looking for role = "cre" or role = c(..., "cre", ...)
            authors = authors_match.group(1)
            # This is a bit fragile but covers standard cases
            cre_match = re.search(
                r'person\("([^"]+)",\s*"([^"]+)",\s*email\s*=\s*"([^"]+)",.*role.*"cre"',
                authors,
            )
            if cre_match:
                name = f"{cre_match.group(1)} {cre_match.group(2)}"
                email = cre_match.group(3)
            else:
                raise ValueError("Could not find Maintainer in DESCRIPTION")
        else:
            raise ValueError("Could not find Maintainer in DESCRIPTION")
    else:
        name = maintainer_match.group(1).strip()
        email = maintainer_match.group(2).strip()

    print(f"Detected Package: {pkg} {version}")
    print(f"Detected Maintainer: {name} <{email}>")

    return pkg, version, name, email


def submit_cran(tarball_path, package_dir):
    pkg, version, name, email = parse_description(package_dir)

    # Read comment
    comment = f"Releasing version {version}"
    if os.path.exists("cran-comments.md"):
        with open("cran-comments.md", "r", encoding="utf-8") as f:
            comment = f.read()

    url = "https://xmpalantir.wu.ac.at/cransubmit/index2.php"

    files = {
        "uploaded_file": (
            os.path.basename(tarball_path),
            open(tarball_path, "rb"),
            "application/x-gzip",
        ),
    }

    data = {
        "pkg_id": "",
        "name": name,
        "email": email,
        "comment": comment,
        "upload": "Upload the package",
    }

    print(f"Submitting {tarball_path} to CRAN (Step 1: Upload)...")
    try:
        session = requests.Session()
        response = session.post(url, files=files, data=data)
        response.raise_for_status()

        # Step 2: Check for confirmation page and extract pkg_id
        if "Please review and submit" in response.text:
            print("Step 1 successful. Proceeding to Step 2 (Confirmation)...")

            # Extract pkg_id
            pkg_id_match = re.search(
                r'name="pkg_id" type="hidden" value="([^"]+)"', response.text
            )
            if not pkg_id_match:
                print("Error: Could not find pkg_id in confirmation page.")
                print(response.text)
                sys.exit(1)

            pkg_id = pkg_id_match.group(1)
            print(f"Found pkg_id: {pkg_id}")

            # Prepare data for Step 2
            confirm_data = {
                "pkg_id": pkg_id,
                "name": name,
                "email": email,
                "policy_check": "1",  # Policy check might be needed here or just implicit
                "submit": "Submit package",
            }

            print("Submitting confirmation...")
            response = session.post(url, data=confirm_data)
            response.raise_for_status()

        # Check for final success message
        if (
            "The following package has been uploaded:" in response.text
            or "The maintainer of this package has been sent an email" in response.text
        ):
            print("Submission successful!")
            print("Check your email for the confirmation link.")
        else:
            print("Submission might have failed. Response content:")
            print(response.text)
            sys.exit(1)

    except Exception as e:
        print(f"Error submitting to CRAN: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python submit_cran.py <tarball_path>")
        sys.exit(1)

    tarball = sys.argv[1]
    # Assuming script is run from project root and package-r is there
    package_dir = "package-r"

    submit_cran(tarball, package_dir)
