# app.py - Root entrypoint for Hugging Face Spaces (Streamlit SDK) & Streamlit Community Cloud
import os
import sys
import runpy

# Ensure repository root, app, and api are in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
for path in [root_dir, os.path.join(root_dir, "app"), os.path.join(root_dir, "api")]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Execute the main fortune teller application
runpy.run_path(os.path.join(root_dir, "app", "fortune_app.py"), run_name="__main__")
