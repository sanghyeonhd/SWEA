# setup.py
# Description: Setup script for the Samsung E&A Digital Twin Welding AI System package.
#              Used for packaging and distribution.

from setuptools import setup, find_packages
import os

# Function to read the requirements.txt file
def parse_requirements(filename="requirements.txt"):
    """Load requirements from a pip requirements file."""
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Function to read the README.md file for long description
def read_readme(filename="README.md"):
    """Read the README file for the long description."""
    with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
        return f.read()

# Project metadata
NAME = "samsung_ena_dt_welding_ai"
VERSION = "0.1.0"  # Initial development version
DESCRIPTION = "Samsung E&A Digital Twin AI System for Welding Process Optimization"
LONG_DESCRIPTION = read_readme()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
AUTHOR = "Samsung E&A AI Team (Placeholder)"
AUTHOR_EMAIL = "ai.contact@samsungena.com (Placeholder)"
URL = "https://github.com/your_org/samsung_ena_dt_welding_ai (Placeholder)" # Project URL
LICENSE = "Proprietary" # Or 'MIT', 'Apache-2.0' etc. if open source

# Define where to find the source code
# Assuming your source code is under a directory like 'src/'
# If your .py files are at the root, use find_packages() without 'where'
# and adjust package_dir accordingly.
# Based on previous discussions, a 'src' directory was proposed.
PACKAGE_DIR = {'': 'src'} # Tells setuptools that packages are under 'src'
PACKAGES = find_packages(where='src') # Find all packages in 'src'

# Entry points for command-line scripts (if any)
# For example, if you want to run SystemManager from the command line
ENTRY_POINTS = {
    'console_scripts': [
        'samsung_welding_dt_start = system_manager:main', # Assuming system_manager.py has a main() function
        # Add other scripts if needed
    ],
}

# Installation requirements
INSTALL_REQUIRES = parse_requirements()

# Classifiers to categorize your project (see https://pypi.org/classifiers/)
CLASSIFIERS = [
    "Development Status :: 3 - Alpha", # Or '4 - Beta', '5 - Production/Stable'
    "Intended Audience :: Developers",
    "Intended Audience :: Manufacturing",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: System :: Hardware :: Hardware Drivers", # If robot/sensor control is deep
    "License :: Other/Proprietary License", # Change if applicable
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Operating System :: OS Independent", # Or specify (e.g., "Operating System :: POSIX :: Linux")
]

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
    package_dir=PACKAGE_DIR,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    entry_points=ENTRY_POINTS,
    classifiers=CLASSIFIERS,
    python_requires='>=3.8', # Specify minimum Python version
    # include_package_data=True, # If you have non-Python files inside your package (e.g., templates, data)
    # package_data={
    #     # Example: 'your_package_name': ['data_files/*.json'],
    # },
    # zip_safe=False, # Usually fine to leave as default (True) unless specific issues
)