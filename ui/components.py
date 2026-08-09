import streamlit as st


def navbar():
    """Top Navigation Bar"""

    col1, col2, col3, col4, col5 = st.columns([5, 1, 1, 1, 1])

    with col1:
        st.markdown(
            """
            <h2 style="margin-top:8px;">
                🌱 AgriSahayak AI
            </h2>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if st.button("Home", use_container_width=True):
            st.session_state.page = "Home"

    with col3:
        if st.button("Chat", use_container_width=True):
            st.session_state.page = "Chat"

    with col4:
        if st.button("About", use_container_width=True):
            st.session_state.page = "About"

    with col5:
        if st.button("Team", use_container_width=True):
            st.session_state.page = "Team"

    st.divider()


def footer():
    st.divider()

    st.markdown(
        """
        <center>

        Made with ❤️ using Streamlit | Retrieval-Augmented Generation (RAG)

        </center>
        """,
        unsafe_allow_html=True,
    )