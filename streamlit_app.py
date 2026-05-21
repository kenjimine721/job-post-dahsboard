"""Public Streamlit Community Cloud entry point.

Streamlit reruns this file on every widget interaction. Execute the dashboard
script each time instead of importing it once and then reusing Python's module
cache on later reruns.
"""

from __future__ import annotations

import runpy


runpy.run_path("app_research_dashboard.py", run_name="__main__")
