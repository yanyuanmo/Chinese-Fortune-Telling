# app/fortune_api_utils.py
"""
API & In-Process utilities for the Fortune Teller frontend.
Supports both:
1. Microservice mode: calls external FastAPI when API_URL is set (e.g. AWS ECS / docker-compose)
2. Direct In-Process mode: calls LangChain RAG directly (e.g. 0-cost Hugging Face Spaces / Streamlit Cloud)
"""

import requests
import streamlit as st
from datetime import datetime
import os
import sys

# Prefer environment variable if set (e.g., from ECS task or local docker-compose)
API_URL = os.getenv("API_URL", "")

def _sync_secrets_to_env():
    """Sync Streamlit secrets to environment variables if available."""
    try:
        if hasattr(st, "secrets"):
            for k in ["MOONSHOT_API_KEY", "moonshot_api_key", "Moonshot_Api_Key", "KIMI_API_KEY"]:
                if k in st.secrets:
                    val = str(st.secrets[k]).strip().strip('"').strip("'")
                    if val:
                        os.environ["MOONSHOT_API_KEY"] = val
                        break
    except Exception:
        pass

def _get_local_fortune_response(question, session_id, model, query_type="general", birth_date=None, birth_gender=None, zodiac_sign=None):
    """Direct in-process execution for serverless/0-cost platforms like Hugging Face Spaces & Streamlit Cloud."""
    _sync_secrets_to_env()
    
    api_key = os.environ.get("MOONSHOT_API_KEY", "").strip()
    if not api_key:
        st.error(
            "⚠️ 未检测到有效配置的 `MOONSHOT_API_KEY`。\n\n"
            "请在 Streamlit 页面右下角点击 **Manage app** ➡️ **⋮** ➡️ **Settings** ➡️ **Secrets**，填入：\n"
            "```toml\n"
            "MOONSHOT_API_KEY = \"你的Kimi密钥\"\n"
            "```\n"
            "保存后刷新页面即可。"
        )
        return None
    
    # Ensure api and root directories are in sys.path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_dir = os.path.join(root_dir, "api")
    for path in [root_dir, api_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        import uuid
        from db_utils import insert_application_logs, get_chat_history, create_application_logs, create_document_store
        from fortune_langchain_utils import get_fortune_chain

        create_application_logs()
        create_document_store()

        actual_session_id = session_id or str(uuid.uuid4())
        chat_history = get_chat_history(actual_session_id)

        # Process input based on query type (matches fortune_main.py)
        if query_type == "bazi":
            prompt_question = f"BaZi analysis for someone born on {birth_date}, gender: {birth_gender or 'not specified'}. {question}"
        elif query_type == "forecast":
            current_year = datetime.now().year
            prompt_question = f"Yearly forecast for {current_year} for {zodiac_sign or 'a person'} with the question: {question}"
        else:
            prompt_question = question

        fortune_chain = get_fortune_chain(query_type=query_type, model=model)
        fortune_result = fortune_chain.invoke({
            "input": prompt_question,
            "chat_history": chat_history
        })

        answer = fortune_result.get('answer', "I cannot see clearly at this moment. Please ask again.")
        insert_application_logs(actual_session_id, prompt_question, answer, model)

        return {
            "answer": answer,
            "session_id": actual_session_id,
            "model": model,
            "query_type": query_type
        }
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Invalid Authentication" in err_msg:
            st.error(
                "⚠️ **Kimi 认证失败 (Error 401 - Invalid Authentication)**\n\n"
                "可能的原因：\n"
                "1. 在 Streamlit 的 **Secrets** 中配置的 `MOONSHOT_API_KEY` 不正确、已过期或复制时包含了多余空格。\n"
                "2. 你的 Kimi (Moonshot AI) 开发者账号额度用尽或已欠费。\n\n"
                "👉 请前往 [Moonshot 开放平台](https://platform.moonshot.cn/console/api-keys) 检查密钥状态或重新生成一个新的 Key。"
            )
        else:
            st.error(f"Error during divination reading: {err_msg}")
        return None

def get_fortune_response(question, session_id, model, query_type="general", birth_date=None, birth_gender=None, zodiac_sign=None):
    """
    Send a fortune telling request.
    Uses HTTP API if API_URL is reachable; otherwise falls back to local direct execution.
    """
    if API_URL:
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = {
            "question": question,
            "model": model,
            "query_type": query_type
        }
        if session_id:
            data["session_id"] = session_id
        if birth_date:
            data["birth_date"] = birth_date
        if birth_gender:
            data["birth_gender"] = birth_gender
        if zodiac_sign:
            data["zodiac_sign"] = zodiac_sign

        try:
            response = requests.post(f"{API_URL}/fortune", headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"Remote API returned status {response.status_code}, falling back to local engine...")
        except Exception:
            # Remote API unavailable, fall back to local direct execution
            pass

    # Direct in-process fallback
    return _get_local_fortune_response(
        question=question,
        session_id=session_id,
        model=model,
        query_type=query_type,
        birth_date=birth_date,
        birth_gender=birth_gender,
        zodiac_sign=zodiac_sign
    )

def get_zodiac_signs():
    """Get the list of Chinese zodiac signs."""
    if API_URL:
        try:
            response = requests.get(f"{API_URL}/zodiac-signs", timeout=3)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass

    return [
        "Rat (鼠)", "Ox (牛)", "Tiger (虎)", "Rabbit (兔)",
        "Dragon (龙)", "Snake (蛇)", "Horse (马)", "Goat (羊)",
        "Monkey (猴)", "Rooster (鸡)", "Dog (狗)", "Pig (猪)"
    ]

def get_fortune_methods():
    """Get the list of fortune telling methods."""
    if API_URL:
        try:
            response = requests.get(f"{API_URL}/fortune-methods", timeout=3)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass

    return [
        {
            "name": "BaZi (八字)",
            "description": "Four Pillars of Destiny based on birth time",
            "query_type": "bazi"
        },
        {
            "name": "Zi Wei Dou Shu (紫微斗数)",
            "description": "Purple Star Astrology for detailed life readings",
            "query_type": "general"
        },
        {
            "name": "Qi Men Dun Jia (奇門遁甲)",
            "description": "Ancient divination system for strategic decisions",
            "query_type": "general"
        },
        {
            "name": "I Ching (易经)",
            "description": "Book of Changes divination using hexagrams",
            "query_type": "general"
        },
        {
            "name": "Yearly Forecast (年运)",
            "description": "Predictions for the coming year based on zodiac sign",
            "query_type": "forecast"
        }
    ]

def calculate_chinese_zodiac(birth_year):
    """Calculate Chinese zodiac sign based on birth year."""
    zodiac_animals = [
        "Rat (鼠)", "Ox (牛)", "Tiger (虎)", "Rabbit (兔)",
        "Dragon (龙)", "Snake (蛇)", "Horse (马)", "Goat (羊)",
        "Monkey (猴)", "Rooster (鸡)", "Dog (狗)", "Pig (猪)"
    ]
    index = (birth_year - 2020) % 12
    if index < 0:
        index += 12
    return zodiac_animals[index]
