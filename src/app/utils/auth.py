import streamlit as st
from utils.config import config

def check_authentication():
    return st.session_state.get('authenticated', False)

def login(username: str, password: str) -> bool:
    if username == config.auth_username and password == config.auth_password:
        st.session_state.authenticated = True
        st.session_state.username = username
        return True
    return False

def logout():
    st.session_state.authenticated = False
    if 'username' in st.session_state:
        del st.session_state.username

def render_login_page():
    st.markdown("""
        <style>
            .login-container {
                max-width: 400px;
                margin: 100px auto;
                padding: 2rem;
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .login-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .login-header h1 {
                color: #1f1f1f;
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }
            .login-header p {
                color: #666;
                font-size: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class="login-header">
                <h1>🔐 Login</h1>
                <p>Document Q&A RAG System</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                elif login(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.markdown("""
            <div style='text-align: center; margin-top: 3rem; padding: 20px 0;'>
                <p style='font-size: 16px; color: #999; margin: 0;'>Powered by</p>
                <p style='font-size: 20px; font-weight: 600; color: #1f1f1f; margin: 5px 0;'>
                    Infoservices
                </p>
                <p style='font-size: 14px; color: #999; margin: 0;'>© 2025 All Rights Reserved</p>
            </div>
        """, unsafe_allow_html=True)
