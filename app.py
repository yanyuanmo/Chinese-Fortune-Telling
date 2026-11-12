# app.py - Root entrypoint for Hugging Face Spaces & Streamlit Community Cloud
import os
import sys
import runpy

# Ensure repository root, app, and api are in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
for path in [root_dir, os.path.join(root_dir, "app"), os.path.join(root_dir, "api")]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Sync secrets to environment variables immediately on startup
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for k in ["MOONSHOT_API_KEY", "moonshot_api_key", "Moonshot_Api_Key", "KIMI_API_KEY"]:
            if k in st.secrets:
                val = str(st.secrets[k]).strip().strip('"').strip("'")
                if val:
                    os.environ["MOONSHOT_API_KEY"] = val
                    break
except Exception:
    pass

# Execute the main fortune teller application
runpy.run_path(os.path.join(root_dir, "app", "fortune_app.py"), run_name="__main__")
