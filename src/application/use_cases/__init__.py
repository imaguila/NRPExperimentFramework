"""
Headless application use cases.

Use cases coordinate existing domain and computational components without
depending on Streamlit or any other user-interface technology.
"""
from src.application.use_cases.load_json_case import load_json_case
from src.application.use_cases.run_optimization import run_optimization

__all__ = [
    "load_json_case",
    "run_optimization",
]