#!/bin/bash
set -e

echo "🔮 Starting FastAPI backend on 127.0.0.1:8000..."
uvicorn fortune_main:app --app-dir api --host 127.0.0.1 --port 8000 &

echo "⏳ Waiting for backend API to initialize..."
until python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').getcode() == 200 else 1)" 2>/dev/null; do
    sleep 1
done
echo "✅ Backend API is ready!"

echo "🚀 Starting Streamlit frontend on 0.0.0.0:7860..."
exec streamlit run app/fortune_app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
