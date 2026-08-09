import streamlit as st

from ui import landing, chat, about, team

st.set_page_config(
    page_title="AgriSahayak AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================
# Load CSS
# ==========================
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================
# Session State
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ==========================
# Navigation Bar
# ==========================
nav1, nav2, nav3, nav4, nav5 = st.columns([4, 1, 1, 1, 1])

with nav1:
    st.markdown(
        """
        <div style="font-size:30px;font-weight:800;">
            AgriSahayak AI
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav2:
    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"

with nav3:
    if st.button("Chat", use_container_width=True):
        st.session_state.page = "Chat"

with nav4:
    if st.button("About", use_container_width=True):
        st.session_state.page = "About"

with nav5:
    if st.button("Team", use_container_width=True):
        st.session_state.page = "Team"

st.divider()

# ==========================
# Routing
# ==========================
if st.session_state.page == "Home":
    landing.show()

elif st.session_state.page == "Chat":
    chat.show()

elif st.session_state.page == "About":
    about.show()

elif st.session_state.page == "Team":
    team.show()