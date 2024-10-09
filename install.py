# This script adds the pth file to the venv site packages to ensure that the library files are available
# NOTE: virtual environment folder is set to .venv. if you specified a different name, please change the path accordingly in the script
import os
import sys

current_lib_location = os.path.join(os.path.dirname(__file__), "qga_lib")

site_package_location = ""
for cur_path in sys.path:
    if os.path.basename(cur_path) == "site-packages":
        site_package_location = os.path.join(cur_path, "qga_lib.pth")

if site_package_location == "":
    raise Exception("Site-package folder not found!")
else:
    print(f"Adding pth file: {site_package_location}")
    

with open(site_package_location, "w") as cur_pth:
    print(current_lib_location)
    cur_pth.write(current_lib_location)