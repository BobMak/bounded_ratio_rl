import os
import glob
import sys

site_packages = os.path.abspath(".venv/lib/python3.10/site-packages")
os.makedirs(site_packages, exist_ok=True)

archive_dir = os.path.expanduser("~/.cache/uv/archive-v0")
archives = glob.glob(os.path.join(archive_dir, "*"))

print(f"Scanning {len(archives)} archives in {archive_dir}...")

# Collect pure python and python-310 specific packages
chosen = {}

for a in archives:
    dist_info = glob.glob(os.path.join(a, "*.dist-info"))
    if not dist_info:
        continue
    pkg_name = os.path.basename(dist_info[0]).split("-")[0].lower()
    
    # Check what .so files are present
    so_files = glob.glob(os.path.join(a, "**/*.so"), recursive=True)
    
    if not so_files:
        # Pure python package - universally valid
        chosen[pkg_name] = a
    else:
        # Has C extensions - must support cpython-310 or abi3
        has_cp310 = any("cpython-310" in s or "abi3" in s for s in so_files)
        if has_cp310:
            chosen[pkg_name] = a

print(f"Found {len(chosen)} compatible packages for Python 3.10.")

linked_count = 0
for pkg, path in chosen.items():
    # Link all items inside path (except dist-info if preferred, or include it)
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(site_packages, item)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
                linked_count += 1
            except Exception as e:
                pass

print(f"Successfully linked {linked_count} entries into {site_packages}")
