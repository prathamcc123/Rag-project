import sys
from pathlib import Path

# --------------------------------------------------
# Fix Backend Imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import sys
from pathlib import Path

# Fix Backend Imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from backend.retrieve import retrieve_query
from backend.generate_answer import generate_answer

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="AgriSahayak AI",
    page_icon="🌾",
    layout="wide"
)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🌾 AgriSahayak AI")

st.caption(
    "Multilingual Agriculture Assistant powered by Fine-Tuned MuRIL and RAG"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("🌾 AgriSahayak AI")
    with st.sidebar:

     st.header("🌾 AgriSahayak AI")

    language = st.selectbox(
        "🌐 Select Language",
        ["Hindi", "English"]
    )
    st.markdown("---")

    st.markdown("---")

    st.subheader("Model Information")

    st.info(
"""
🌾 Agriculture Knowledge Base

📚 7808 Documents

🤖 Fine-Tuned MuRIL

⚡ FAISS Vector Search

🌐 Hindi + English Queries
"""
)

    st.markdown("---")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if len(st.session_state.messages) == 0:

         st.markdown(
        """
        ###  Welcome to AgriSahayak AI

        Ask questions about:

        🌾 Crop Cultivation

        💧 Irrigation

        🌱 Organic Farming

        🐛 Pest Management

        🧪 Fertilizers & Soil Health
        """
    )

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

prompt = st.chat_input(
    "कृषि से संबंधित प्रश्न पूछें..."
)

if prompt:

    # User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
with st.chat_message("assistant"):

        try:

            with st.spinner("🔍 Searching knowledge base..."):

                retrieved_chunks, context = retrieve_query(prompt)

                answer = generate_answer(
                    query=prompt,
                    context=context,
                )
                from deep_translator import GoogleTranslator

                if language == "English":

                    try:

                        answer = GoogleTranslator(
                            source="hi",
                            target="en"
                        ).translate(answer)

                    except Exception:
                        pass

                st.markdown(answer)

                with st.expander("📚 Retrieved Sources"):
                    

                    for i, chunk in enumerate(retrieved_chunks, start=1):
                        st.markdown(
                          f"""
                           **{i}. {chunk.get('title','Untitled')}**

                           Score: {chunk.get('score',0):.3f}

                          {chunk.get('summary','')}
                          """
    )
                        st.divider()
    
        except Exception as e:

            answer = f"❌ Error: {str(e)}"
            st.error(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------


st.markdown("---")

st.caption(
    "Powered by Fine-Tuned MuRIL + FAISS Retrieval"
)