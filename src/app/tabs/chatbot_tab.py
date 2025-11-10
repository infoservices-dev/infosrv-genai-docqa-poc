import streamlit as st
import requests
import json
import time
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_API_CONFIG = {
    "endpoint": os.environ.get("API_ENDPOINT", ""),
    "api_key": os.getenv("API_KEY", ""),
    "state_machine_arn": os.environ.get("STATE_MACHINE_ARN", "")
}

def render_chatbot_tab():
    st.header("🤖 AI Document Assistant")
    
    st.markdown("""
    <style>
    .loading-spinner {
        display: inline-block;
        width: 30px;
        height: 30px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #1f77b4;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    with st.expander("🔧 API Configuration", expanded=False):
        api_endpoint, api_key, state_machine_arn = render_api_config()
    
    if st.session_state.pending_query and st.session_state.is_processing:
        query_data = st.session_state.pending_query
        st.session_state.pending_query = None
        process_query(query_data['query'], query_data['api_endpoint'], 
                     query_data['api_key'], query_data['state_machine_arn'])
    
    render_suggested_questions(api_endpoint, api_key, state_machine_arn)
    
    st.subheader("💬 Chat")
    
    for message in st.session_state.chat_history:
        render_message(message)
    
    if st.session_state.is_processing:
        render_loading()
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_query = st.text_input(
                "Ask a question...",
                placeholder="e.g., What are the key features?",
                disabled=st.session_state.is_processing,
                label_visibility="collapsed"
            )
        
        with col2:
            send_text = "⏳ Processing..." if st.session_state.is_processing else "Send"
            send_button = st.form_submit_button(
                send_text, 
                type="primary", 
                disabled=st.session_state.is_processing,
                use_container_width=True
            )
        
        if send_button and user_query.strip() and not st.session_state.is_processing:
            queue_query(user_query.strip(), api_endpoint, api_key, state_machine_arn)
            st.rerun()
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear", disabled=st.session_state.is_processing):
            st.session_state.chat_history = []
            st.rerun()

def init_session_state():
    defaults = {
        'is_processing': False,
        'processing_start_time': None,
        'suggested_questions': [],
        'questions_loaded': False,
        'chat_history': [],
        'pending_query': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_api_config():
    col1, col2 = st.columns(2)
    with col1:
        api_endpoint = st.text_input("API Endpoint", value=DEFAULT_API_CONFIG["endpoint"])
        api_key = st.text_input("API Key", value=DEFAULT_API_CONFIG["api_key"], type="password")
    with col2:
        state_machine_arn = st.text_area("State Machine ARN", value=DEFAULT_API_CONFIG["state_machine_arn"], height=80)
        
        if st.button("🔍 Test"):
            response = call_api("Hello", api_endpoint, api_key, state_machine_arn)
            if response.get('success'):
                st.success("✅ Connected")
            else:
                st.error(f"❌ Failed: {response.get('error', 'Unknown')}")
    
    return api_endpoint, api_key, state_machine_arn

def render_suggested_questions(api_endpoint, api_key, state_machine_arn):
    if not st.session_state.questions_loaded and not st.session_state.suggested_questions:
        load_questions(api_endpoint, api_key, state_machine_arn)
    
    if st.session_state.suggested_questions:
        with st.expander("💡 Suggested Questions", expanded=True):
            col1, col2 = st.columns(2)
            
            for i, question in enumerate(st.session_state.suggested_questions):
                column = col1 if i % 2 == 0 else col2
                with column:
                    if st.button(f"❓ {question}", key=f"q_{i}", 
                               disabled=st.session_state.is_processing,
                               use_container_width=True):
                        queue_query(question, api_endpoint, api_key, state_machine_arn)
                        st.rerun()

def load_questions(api_endpoint, api_key, state_machine_arn):
    with st.spinner("Loading questions..."):
        try:
            response = call_api(
                "Provide 6 diverse sample questions about the documents. Return only questions, one per line.",
                api_endpoint, api_key, state_machine_arn
            )
            
            if response.get('success'):
                questions = parse_questions(response.get('answer', ''))
                st.session_state.suggested_questions = questions if questions else get_fallback_questions()
            else:
                st.session_state.suggested_questions = get_fallback_questions()
        except Exception:
            st.session_state.suggested_questions = get_fallback_questions()
        
        st.session_state.questions_loaded = True

def parse_questions(text):
    questions = []
    for line in text.split('\n'):
        line = line.strip()
        if len(line) < 10:
            continue
        
        for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '•', '-', '*']:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        
        if line.endswith('?') or any(w in line.lower() for w in ['what', 'how', 'when', 'where', 'why']):
            if len(line) <= 100:
                questions.append(line)
    
    return questions[:6]

def get_fallback_questions():
    return [
        "What documents are available?",
        "Summarize the insurance policy?",
        "What are the coverage details?",
        "How do I file a claim?",
        "What are the payment terms?",
        "Are there any exclusions?"
    ]

def call_api(query, api_endpoint, api_key, state_machine_arn, timeout=60):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    
    payload = {
        "stateMachineArn": state_machine_arn,
        "input": json.dumps({"query": query})
    }
    
    try:
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'output' in result and result['output']:
                output_data = json.loads(result['output'])
                
                return {
                    'success': True,
                    'answer': output_data.get('final_answer', 'No answer'),
                    'citations': output_data.get('citations', []),
                    'routing_category': output_data.get('routing_category'),
                    'categories': output_data.get('categories', [])
                }
            
            return {
                'success': False,
                'error': 'No output received',
                'answer': 'System did not return a response.'
            }
        
        return {
            'success': False,
            'error': f'Status {response.status_code}',
            'answer': f'Request failed with status {response.status_code}'
        }
    
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Timeout', 'answer': 'Request timed out.'}
    except Exception as e:
        return {'success': False, 'error': str(e), 'answer': f'Error: {str(e)}'}

def render_message(message):
    if message['type'] == 'user':
        with st.chat_message("user"):
            st.write(f"**You** · {message['timestamp']}")
            st.write(message['content'])
    else:
        with st.chat_message("assistant"):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**Assistant** · {message['timestamp']}")
            with col2:
                if 'response_time' in message:
                    st.caption(f"⏱️ {message['response_time']}s")
            
            st.write(message['content'])
            
            if message.get('success', True):
                if message.get('citations'):
                    unique_sources = {}
                    for citation in message['citations']:
                        source_uri = citation.get('source_uri', 'Unknown')
                        if source_uri not in unique_sources:
                            unique_sources[source_uri] = citation
                    
                    with st.expander(f"📎 Sources ({len(unique_sources)})"):
                        for i, (source_uri, citation) in enumerate(unique_sources.items(), 1):
                            st.write(f"**[{i}]** {source_uri}")
                
                if message.get('routing_category'):
                    st.info(f"🎯 {message['routing_category'].replace('_', ' ').title()}")
            else:
                if message.get('error'):
                    st.error(f"**Error:** {message['error']}")

def render_loading():
    with st.chat_message("assistant"):
        elapsed = 0
        if st.session_state.processing_start_time:
            elapsed = int(time.time() - st.session_state.processing_start_time)
        
        st.write("**Assistant** · Processing")
        st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
        
        if elapsed > 0:
            st.caption(f"⏱️ {elapsed}s")

def queue_query(query, api_endpoint, api_key, state_machine_arn):
    st.session_state.is_processing = True
    st.session_state.processing_start_time = time.time()
    
    st.session_state.chat_history.append({
        'type': 'user',
        'content': query,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    st.session_state.pending_query = {
        'query': query,
        'api_endpoint': api_endpoint,
        'api_key': api_key,
        'state_machine_arn': state_machine_arn
    }

def process_query(query, api_endpoint, api_key, state_machine_arn):
    try:
        start_time = time.time()
        response = call_api(query, api_endpoint, api_key, state_machine_arn)
        response_time = time.time() - start_time
        
        st.session_state.chat_history.append({
            'type': 'assistant',
            'content': response.get('answer', 'No response'),
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'success': response.get('success', False),
            'response_time': round(response_time, 2),
            'citations': response.get('citations', []),
            'routing_category': response.get('routing_category'),
            'error': response.get('error') if not response.get('success') else None
        })
    except Exception as e:
        st.session_state.chat_history.append({
            'type': 'assistant',
            'content': f'Error: {str(e)}',
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'success': False,
            'error': str(e)
        })
    finally:
        st.session_state.is_processing = False
        st.session_state.processing_start_time = None
        st.rerun()
