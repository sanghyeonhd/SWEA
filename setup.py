# setup.py
# Description: (MODIFIED) Setup script for the ADVANCED Samsung E&A Digital Twin Welding AI System package.

from setuptools import setup, find_packages
import os

# --- Helper Functions ---
def parse_requirements(filename="requirements.txt"):
    """Load requirements from a pip requirements file."""
    # Ensure requirements.txt is in the same directory as setup.py
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(req_path):
        print(f"Warning: {filename} not found. No dependencies will be installed via this setup.")
        return []
    with open(req_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def read_file(filename):
    """Read a file's content."""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(file_path):
        print(f"Warning: File {filename} not found. Content will be empty.")
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# --- Project Metadata ---
NAME = "samsung_ena_dt_welding_ai"
VERSION = "0.2.0"  # Incremented version for advanced features
DESCRIPTION = "Samsung E&A Digital Twin AI System for Welding Process Optimization (Advanced)"
LONG_DESCRIPTION = read_file("README.md") # Assumes README.md is in the project root
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
AUTHOR = "이재림 (자동화부서)" # Updated author
AUTHOR_EMAIL = "jaerim.lee@samsungena.com (Placeholder)" # Placeholder email
URL = "https://your.internal.samsung.ena.portal/dt_welding_ai (Placeholder)" # Internal project URL
LICENSE = "Proprietary - Samsung E&A Internal Use Only" # Updated license

# Define where to find the source code (assuming 'src/' directory)
PACKAGE_DIR = {'': 'src'}
PACKAGES = find_packages(where='src', exclude=['tests*', '*.tests', '*.tests.*', 'tests'])
# find_packages will automatically find sub-packages within src if they have __init__.py

# --- Entry Points for Command-Line Scripts ---
# This allows running the system manager directly from the command line after installation.
# Ensure system_manager.py has a main() function.
ENTRY_POINTS = {
    'console_scripts': [
        'sena_welding_dt_manager = system_manager:main', # Example command to start the system
        # Add other entry points if needed, e.g., for trainer, data generation scripts
        # 'sena_welding_dt_train = trainer:run_training',
        # 'sena_welding_dt_dummy_data_gen = create_dummy_data_main_script:main', # Conceptual
    ],
}

# --- Installation Requirements ---
# Reads from requirements.txt. Ensure all new dependencies (pika, psycopg2-binary, dask, onnxruntime, etc.)
# are listed in requirements.txt.
INSTALL_REQUIRES = parse_requirements()

# --- Classifiers ---
# See https://pypi.org/classifiers/
CLASSIFIERS = [
    "Development Status :: 3 - Alpha", # Or "4 - Beta" as features are added
    "Intended Audience :: Developers",
    "Intended Audience :: Manufacturing",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: System :: Distributed Computing",
    "Topic :: System :: Monitoring",
    "Topic :: System :: Networking",
    "License :: Other/Proprietary License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Operating System :: POSIX :: Linux", # Assuming Linux for many industrial/server deployments
    "Operating System :: Microsoft :: Windows", # If also developed/run on Windows
]

# --- Main Setup Call ---
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,

    package_dir=PACKAGE_DIR, # Specifies that packages are in 'src'
    packages=PACKAGES,       # Finds all packages under 'src'

    install_requires=INSTALL_REQUIRES, # Dependencies from requirements.txt
    entry_points=ENTRY_POINTS,         # Command-line scripts

    python_requires='>=3.8', # Minimum Python version

    # --- Including Non-Python Files within your Package ---
    # If you have config files (e.g., robot_configs.json, adaptive_rules.json in src/config_files/),
    # or other data files that need to be installed with your package, use include_package_data
    # and either MANIFEST.in or package_data.
    include_package_data=True, # Tells setuptools to look at MANIFEST.in or VCS for data files

    # package_data is an alternative to MANIFEST.in for simpler cases.
    # Example: if you have a 'src/your_package_name/config_files/*.json'
    # package_data={
    #     'your_package_name': ['config_files/*.json', 'assets/*.png'],
    # },
    # For `src` layout where `your_package_name` is the directory inside `src` with `__init__.py`:
    # If `system_manager` is a top-level module in `src` (i.e., `src/system_manager.py`),
    # and you want to include a `src/default_configs/*.json` directory:
    # package_data={
    #      # This key is the package name as Python sees it.
    #      # If your modules are directly under src, there's no single "package name" at root of src
    #      # unless src itself is the package.
    #      # Usually, you'd have `src/samsung_ena_dt_welding_ai/` and then modules inside that.
    #      # For now, assuming a MANIFEST.in is preferred for flexibility.
    # },

    # A MANIFEST.in file is often more flexible for including data files.
    # Example MANIFEST.in content (place this file in the project root):
    #
    # include README.md
    # include requirements.txt
    # recursive-include src/config_files *.json  # Example: include all json from a config_files dir in src
    # recursive-include data *.csv             # Example: if you wanted to package example data
    # recursive-include models *.pth *.pkl     # Example: if you wanted to package dummy models

    classifiers=CLASSIFIERS,
    zip_safe=False, # Set to False if your package relies on file system paths or C extensions
                    # that might not work well when zipped. Often safer to set to False.
)

# --- Post-setup message (optional) ---
print(f"\n--- {NAME} version {VERSION} setup complete ---")
print("To install in editable mode for development: pip install -e .")
print("To build distributable packages: python setup.py sdist bdist_wheel")