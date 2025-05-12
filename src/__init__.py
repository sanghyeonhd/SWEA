# src/__init__.py
# This file makes the 'src' directory a Python package.

# You can define package-level variables, e.g., version
# __version__ = "0.1.0"

# You can also make specific modules or sub-packages available directly
# when the 'src' package is imported, though this is often kept minimal
# to avoid circular imports or overly eager loading.

# Example: If you want to allow `from src import SystemManager`
# instead of `from src.system_manager import SystemManager`
# (assuming system_manager.py is directly under src/)

# from .system_manager import SystemManager
# from .ai_model import WeldingAIModel
# from .data_logger_db import DataLoggerDB
# ... and so on for other key classes or functions you want to expose

# For now, keeping it simple.
# The presence of this file is enough to make 'src' a package.

# It's also common to use this file to control what `from src import *` does,
# by defining `__all__`.
# __all__ = ["SystemManager", "WeldingAIModel", "DataLoggerDB"] # Example