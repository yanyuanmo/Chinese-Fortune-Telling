# app/fortune_sidebar.py
"""
Sidebar component for the Fortune Teller application.
"""

import streamlit as st
from datetime import datetime
from fortune_api_utils import get_zodiac_signs, get_fortune_methods, calculate_chinese_zodiac

def display_fortune_sidebar():
    """Display the sidebar with fortune telling options."""
    # Add decorative header
    st.sidebar.markdown("# 🔮 果赖算命阁 🔮")
    st.sidebar.markdown("### Master Guo Lai's Fortune Hall")
    
    # Model Selection (Current active Moonshot Kimi models)
    model_options = [
        "kimi-k3.1-pro",
        "kimi-k3.1-pro-highspeed",
        "kimi-k3"
    ]
    st.sidebar.selectbox(
        "Select Oracle Engine", 
        options=model_options, 
        key="model",
        help="Select the divination engine to power your reading"
    )
    
    # Divination Method
    st.sidebar.markdown("## 🏮 Divination Method 🏮")
    methods = get_fortune_methods()
    method_options = {method["name"]: method["query_type"] for method in methods}
    
    selected_method_name = st.sidebar.selectbox(
        "Select Method", 
        options=list(method_options.keys()),
        key="fortune_method_name"
    )
    
    # Store the query type based on selected method
    st.session_state.query_type = method_options[selected_method_name]
    
    # Show description of selected method
    selected_method_info = next((m for m in methods if m["name"] == selected_method_name), None)
    if selected_method_info:
        st.sidebar.markdown(f"*{selected_method_info['description']}*")
    
    # Additional inputs based on divination method
    if st.session_state.query_type == "bazi":
        st.sidebar.markdown("## 📅 Birth Information 📅")
        
        # Birth date picker
        birth_date = st.sidebar.date_input(
            "Birth Date",
            datetime(1990, 1, 1),
            key="birth_date"
        )
        
        # Birth time picker
        birth_time = st.sidebar.time_input(
            "Birth Time (if known)",
            datetime(2020, 1, 1, 12, 0),
            key="birth_time"
        )
        
        # Combine date and time
        st.session_state.birth_datetime = f"{birth_date} {birth_time.strftime('%H:%M')}"
        
        # Gender selection
        gender_options = ["Not specified", "Male", "Female"]
        selected_gender = st.sidebar.radio("Gender", gender_options, key="gender_widget")
        st.session_state.birth_gender = None if selected_gender == "Not specified" else selected_gender.lower()
                
        # Calculate and display Chinese zodiac sign
        birth_year = birth_date.year
        zodiac_sign = calculate_chinese_zodiac(birth_year)
        st.sidebar.markdown(f"**Chinese Zodiac:** {zodiac_sign}")
        st.session_state.zodiac_sign = zodiac_sign
        
    elif st.session_state.query_type == "forecast":
        st.sidebar.markdown("## 🐲 Chinese Zodiac 🐲")
        
        # Zodiac sign selection
        zodiac_signs = get_zodiac_signs()
        selected_zodiac = st.sidebar.selectbox(
            "Select Your Chinese Zodiac Sign",
            options=zodiac_signs,
            key="zodiac_sign"
        )
        st.session_state.zodiac_sign = selected_zodiac
        
        # Current year
        current_year = datetime.now().year
        st.sidebar.markdown(f"**Forecast Year:** {current_year}")
    
    # Divination tips accordion
    with st.sidebar.expander("🧙 Divination Tips 🧙"):
        st.markdown("""
        - Be clear and specific in your questions
        - Focus on one issue at a time
        - Avoid yes/no questions for deeper insights
        - Be open to symbolic and indirect answers
        - Reflect on the advice received before acting
        """)
    
    # Reset button
    if st.sidebar.button("🔄 Begin New Reading"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()