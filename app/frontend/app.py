import os
import re

import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://backend:8000",
)

API_URL = f"{BACKEND_URL}/ask"


st.set_page_config(
    page_title="LegalAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       APP
    ======================================================== */

    .stApp {
        background: #0b1120;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }


    /* ========================================================
       HEADINGS
    ======================================================== */

    h1 {
        font-size: 38px !important;
        font-weight: 700 !important;
        letter-spacing: -1px;
    }

    h3 {
        color: #e5e7eb !important;
    }


    /* ========================================================
       SUBTITLE
    ======================================================== */

    .subtitle-text {
        color: #94a3b8;
        font-size: 15px;
        margin-top: -12px;
        margin-bottom: 25px;
    }


    /* ========================================================
       STATUS
    ======================================================== */

    .status-box {
        background: #123524;
        border: 1px solid #1d6b46;
        border-radius: 20px;
        padding: 7px 13px;
        color: #6ee7a0;
        display: inline-block;
        font-size: 13px;
        margin-bottom: 20px;
    }


    /* ========================================================
       SIDEBAR METRICS
    ======================================================== */

    .metric-box {
        background: #0f172a;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 12px;
    }

    .metric-number {
        color: #f8fafc;
        font-size: 22px;
        font-weight: 700;
        margin-top: 3px;
    }


    /* ========================================================
       QUESTION AREA
    ======================================================== */

    [data-testid="stTextArea"] textarea {
        background: #1f2029;
        color: #f8fafc;
        border: 1px solid #303746;
        border-radius: 12px;
        font-size: 15px;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: #ff4b4b;
        box-shadow: 0 0 0 1px #ff4b4b;
    }


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 500;
    }

    .stButton > button:hover {
        border-color: #ff4b4b;
        color: #ffffff;
    }


    /* ========================================================
       PRIMARY BUTTON
    ======================================================== */

    button[kind="primary"] {
        min-height: 48px;
        font-size: 15px;
    }


    /* ========================================================
       ANSWER
    ======================================================== */

    .answer-box {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .answer-header {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .answer-text {
        color: #e5e7eb;
        line-height: 1.7;
        font-size: 15px;
    }


    /* ========================================================
       USER QUESTION
    ======================================================== */

    .question-box {
        background: #0f172a;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 15px;
    }

    .question-label {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 5px;
    }

    .question-text {
        color: #f8fafc;
        font-size: 15px;
        line-height: 1.6;
    }


    /* ========================================================
       SIDEBAR PIPELINE
    ======================================================== */

    .pipeline-item {
        padding: 7px 0;
        color: #dbeafe;
        font-size: 14px;
    }

    .pipeline-check {
        color: #6ee7a0;
        font-weight: 700;
    }


    /* ========================================================
       REMOVE STREAMLIT EXTRA SPACING
    ======================================================== */

    [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# ⚖️ LegalAI"
    )

    st.caption(
        "AI-Powered Legal Assistant"
    )

    st.markdown(
        '<div class="status-box">● System Online</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Knowledge Base")

    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-label">
                Legal Documents
            </div>
            <div class="metric-number">
                7
            </div>
        </div>

        <div class="metric-box">
            <div class="metric-label">
                Indexed Pages
            </div>
            <div class="metric-number">
                858
            </div>
        </div>

        <div class="metric-box">
            <div class="metric-label">
                Usable Chunks
            </div>
            <div class="metric-number">
                1,538
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Retrieval Pipeline")

    pipeline = [
        "Vector Search",
        "BM25 Search",
        "Hybrid Retrieval",
        "Cross-Encoder Reranking",
        "Context Verification",
        "Mistral Generation",
    ]

    for item in pipeline:

        st.markdown(
            f"""
            <div class="pipeline-item">
                <span class="pipeline-check">✓</span>
                {item}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button(
        "🗑 Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.selected_question = ""
        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.title("LegalAI")

st.markdown(
    '<p class="subtitle-text">Ask questions and get AI-powered answers from your legal document collection.</p>',
    unsafe_allow_html=True,
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown("### Try a question")

example_cols = st.columns(3)

examples = [
    "What are the principles governing bail?",
    "What safeguards exist during arrest?",
    "What is the role of Article 21?",
]

for col, example in zip(example_cols, examples):

    if col.button(
        example,
        use_container_width=True,
    ):
        st.session_state.selected_question = example
        st.rerun()


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Legal Question",
    value=st.session_state.selected_question,
    height=110,
    placeholder="Ask LegalAI a question about your legal documents...",
    label_visibility="collapsed",
)


# ============================================================
# ASK LEGALAI BUTTON
# ============================================================

ask = st.button(
    "🔍 Ask LegalAI",
    type="primary",
    use_container_width=True,
)


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            """
            <div class="question-box">
                <div class="question-label">
                    You
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            message["content"]
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="answer-box">
                <div class="answer-header">
                    ⚖️ LegalAI
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            message["content"]
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# API REQUEST
# ============================================================

if ask:

    clean_question = question.strip()

    if not clean_question:

        st.warning(
            "Please enter a legal question."
        )

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": clean_question,
            }
        )

        with st.spinner(
            "LegalAI is searching the legal knowledge base..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "query": clean_question
                    },
                    timeout=180,
                )

                response.raise_for_status()

                data = response.json()

                answer = data.get(
                    "answer",
                    "",
                )

                answer = str(answer).strip()

                # ------------------------------------------------
                # Remove accidental HTML tags from LLM response
                # ------------------------------------------------

                answer = re.sub(
                    r"<[^>]+>",
                    "",
                    answer,
                )

                # ------------------------------------------------
                # Decode common HTML entities
                # ------------------------------------------------

                answer = (
                    answer
                    .replace("&nbsp;", " ")
                    .replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                )

                answer = answer.strip()

                if not answer:
                    answer = "No answer was returned."

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.session_state.selected_question = ""

                st.rerun()


            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server."
                )

                st.caption(
                    f"Backend: {API_URL}"
                )


            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. Please try again."
                )


            except requests.exceptions.HTTPError as e:

                st.error(
                    f"FastAPI returned an error: {e}"
                )


            except requests.exceptions.RequestException as e:

                st.error(
                    f"API request failed: {e}"
                )


            except ValueError:

                st.error(
                    "FastAPI returned an invalid response."
                )
                