import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from vector import retriever

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="AI Policy ChatBot", layout="wide")

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
    <style>
    body {
        background-color: #111;
        color: white;
        font-family: 'Inter', sans-serif;
        overflow-x: hidden;
    }
    .main {
        background-color: #111;
        color: white;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1b1b1b;
        color: white;
    }

    /* Chat area */
    .chat-area {
    background-color: #111;
    padding: 10px 40px 20px 40px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;  /* messages start at top */
    gap: 10px;                /* space between messages */
    overflow-y: auto;
    border-radius: 8px;
}


    /* Chat message rows */
    .msg-row { display: flex; margin: 8px 0; }
    .msg-row.user { justify-content: flex-end; }
    .msg-row.bot { justify-content: flex-start; }

    /* Message bubbles */
    .user-bubble {
        background-color: #2f2f2f;
        padding: 12px 16px;
        border-radius: 15px;
        max-width: 65%;
        text-align: right;
        word-wrap: break-word;
        margin-top: 0;  /* remove top space */
    }

    .bot-bubble {
        background-color: #1f1f1f;
        padding: 12px 16px;
        border-radius: 15px;
        border-left: 4px solid #00ADB5;
        max-width: 75%;
        word-wrap: break-word;
        margin-top: 2px;
    }

    /* Remove extra Streamlit spacing */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

# ---------------------------------------------------------
# SIDEBAR - HISTORY (QUESTIONS ONLY)
# ---------------------------------------------------------
st.sidebar.title("📜 Chat History")

# Clear history button
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history.clear()
    st.session_state.selected_index = None
    st.rerun()

# List user questions as clickable buttons
for i, chat in enumerate(st.session_state.history):
    if st.sidebar.button(f"🗨️ {chat['user']}", key=f"q_{i}"):
        st.session_state.selected_index = i
        st.rerun()

# ---------------------------------------------------------
# LOAD MODEL + RETRIEVER
# ---------------------------------------------------------
@st.cache_resource
def load_chain():
    model = OllamaLLM(model="llama3.2")
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain
qa_chain = load_chain()

# ---------------------------------------------------------
# MAIN CHAT DISPLAY
# ---------------------------------------------------------
chat_container = st.container()
with chat_container:
    st.markdown("<div class='chat-area' id='chat-box'>", unsafe_allow_html=True)

    # If user clicked a history question → show only that Q&A
    if st.session_state.selected_index is not None:
        chat = st.session_state.history[st.session_state.selected_index]
        st.markdown(f"<div class='msg-row user'><div class='user-bubble'>{chat['user']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='msg-row bot'><div class='bot-bubble'><b>🤖 Answer:</b><br>{chat['bot']}</div></div>", unsafe_allow_html=True)
    else:
        # Otherwise show entire chat
        for chat in st.session_state.history:
            st.markdown(f"<div class='msg-row user'><div class='user-bubble'>{chat['user']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='msg-row bot'><div class='bot-bubble'><b>🤖 Answer:</b><br>{chat['bot']}</div></div>", unsafe_allow_html=True)

    # Auto-scroll JS
    st.markdown("""
        <script>
        var chatBox = document.getElementById('chat-box');
        if (chatBox) { chatBox.scrollTop = chatBox.scrollHeight; }
        </script>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT AREA (st.chat_input stays pinned at bottom)
# ---------------------------------------------------------
prompt = st.chat_input("Type your question here...")

# ---------------------------------------------------------
# LOGIC FOR ANSWERING
# ---------------------------------------------------------
if prompt:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": prompt})
        answer = result["result"]

    # Add to history
    st.session_state.history.append({"user": prompt, "bot": answer})
    st.session_state.selected_index = len(st.session_state.history) - 1
    st.rerun()
