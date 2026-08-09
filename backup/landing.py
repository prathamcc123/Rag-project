import streamlit as st


def show():

    # ==========================
    # HERO
    # ==========================

    st.markdown(
        """
        <div class="hero">

            <div class="small-heading">
                RETRIEVAL • AUGMENTED • GENERATION
            </div>

            <h1>
                AgriSahayak AI
            </h1>

            <p>
                An intelligent agriculture assistant powered by a fine-tuned
                multilingual sentence encoder, FAISS Vector Search and
                Google Gemini to deliver fast, relevant and trustworthy
                answers for Indian agriculture.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([2, 1, 2])

    with center:
        st.button(
            " Get Started",
            use_container_width=True,
            type="primary",
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ==========================
    # FEATURES
    # ==========================

    st.markdown(
        '<div class="section-title">Why AgriSahayak AI?</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-title">
                    AI Retrieval
                </div>

                <div class="feature-text">
                    Uses a fine-tuned multilingual sentence transformer with
                    FAISS vector search to retrieve the most relevant
                    agricultural knowledge before generating an answer.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-title">
                    Trusted Knowledge
                </div>

                <div class="feature-text">
                    Built on curated Indian agriculture resources that help
                    reduce hallucinations and improve response quality through
                    Retrieval-Augmented Generation.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-title">
                    Multilingual Support
                </div>

                <div class="feature-text">
                    Supports multiple Indian languages using a domain-aware
                    multilingual retrieval model trained for agricultural
                    question answering.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ==========================
    # WORKFLOW
    # ==========================

    st.markdown(
        '<div class="section-title">How It Works</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="step-card">
                <h4>Question</h4>
                User asks an agriculture question.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="step-card">
                <h4>Retrieve</h4>
                FAISS searches the most relevant knowledge chunks.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="step-card">
                <h4>Generate</h4>
                Gemini receives the retrieved context and generates the answer.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="step-card">
                <h4>Answer</h4>
                The user receives an accurate and context-aware response.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ==========================
    # PROJECT STATS
    # ==========================

    st.markdown(
        '<div class="section-title">Project Highlights</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)

    stats = [
        ("7808", "Knowledge Chunks"),
        ("768", "Embedding Size"),
        ("FAISS", "Vector Database"),
        ("Gemini", "LLM"),
    ]

    for col, (number, label) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-number">{number}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ==========================
    # FOOTER
    # ==========================

    st.markdown(
        """
        <div class="footer">

        <h3>AgriSahayak AI</h3>

        <p>
        Retrieval-Augmented Generation for Indian Agriculture
        </p>

        <p>
        Sentence Transformers • FAISS • Gemini • Streamlit
        </p>

        <p>
        © 2026 AgriSahayak AI
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )