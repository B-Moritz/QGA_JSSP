# This script adds the pth file to the venv site packages to ensure that the library files are available
# NOTE: virtual environment folder is set to .venv. if you specified a different name, please change the path accordingly in the script
import os

current_lib_location = os.path.join(os.path.dirname(__file__), "lib")
site_package_location = os.path.join(os.path.dirname(__file__), ".venv\Lib\site-packages\qga_lib.pth")

with open(site_package_location, "w") as cur_pth:
    cur_pth.write(current_lib_location)