import streamlit as st
import os

# 1. Import config constants
from clinical_rag.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    TOP_K,
    LLM_REPO_ID,
    HF_TOKEN,
)

# 2. Import actual retrieval function
from clinical_rag.retrieval import agentic_retrieve

# 3. Import generation function
from clinical_rag.generation import generate_answer

# Page configuration
st.set_page_config(
    page_title="Clinical Agentic RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CACHED RESOURCES TO PREVENT MEMORY SPIKES & CONNECTION RESETS
# -----------------------------------------------------------------------------
@st.cache_resource
def cached_agentic_retrieve(query, top_k):
    """Cache retrieval execution to prevent redundant heavy computations."""
    return agentic_retrieve(query, top_k=top_k, verbose=False)

@st.cache_resource
def cached_generate_answer(query, retrieved_docs):
    """Cache answer generation step if identical query/context occurs."""
    return generate_answer(query, retrieved_docs)

# Inject Custom CSS for Banner, Custom Bullets, and UI Polish
st.markdown("""
<style>
    /* 1. TOP BACKGROUND BANNER */
    .banner-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0284C7 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: #FFFFFF !important;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.3);
        position: relative;
        overflow: hidden;
    }
    .banner-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #FFFFFF !important;
    }
    .banner-subtitle {
        font-size: 1.05rem;
        color: #93C5FD !important;
        font-weight: 400;
        margin-bottom: 0;
    }

    /* 2. CUSTOM BULLET POINTS IN CHAT & ANSWERS */
    [data-testid="stChatMessage"] ul {
        list-style: none !important;
        padding-left: 0 !important;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    [data-testid="stChatMessage"] ul li {
        position: relative;
        padding-left: 2rem !important;
        margin-bottom: 0.75rem !important;
        line-height: 1.6;
        background: #F8FAFC !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0;
        padding: 0.6rem 1rem 0.6rem 2.4rem !important;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    [data-testid="stChatMessage"] ul li::before {
        content: "🔹";
        position: absolute;
        left: 0.75rem;
        top: 0.55rem;
        font-size: 0.95rem;
    }

    /* 3. BADGES FOR SOURCES */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 10px;
    }
    .badge-clinical {
        background-color: #E0F2FE !important;
        color: #0369A1 !important;
        border: 1px solid #BAE6FD;
    }
    .badge-fda {
        background-color: #FFE4E6 !important;
        color: #BE123C !important;
        border: 1px solid #FECDD3;
    }
    .badge-pubmed {
        background-color: #D1FAE5 !important;
        color: #047857 !important;
        border: 1px solid #A7F3D0;
    }

    /* 4. RETRIEVED EVIDENCE CARDS & HEADINGS */
    .evidence-card {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-left: 4px solid #2563EB;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    .evidence-card strong, 
    .evidence-card small, 
    .evidence-card p {
        color: #0F172A !important;
    }
    .evidence-card code {
        color: #1E40AF !important;
        background-color: #EFF6FF !important;
    }

    /* 5. EXPANDEE & HEADINGS COLOR OVERRIDE */
    div[data-testid="stExpander"] h3,
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span {
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP HERO BANNER WITH ICONS
# -----------------------------------------------------------------------------
st.markdown("""
<div class="banner-container">
    <div class="banner-title">
        <span>🩺</span> Clinical Trial Evidence Synthesis Assistant
    </div>
    <div class="banner-subtitle">
        ⚡ Multi-agent evidence retrieval across <strong>PubMed</strong> 📚, <strong>FDA Alerts</strong> 🚨, and <strong>ClinicalTrials.gov</strong> 🧪
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function to render color-coded badges with icons
def render_source_badge(source_name: str) -> str:
    s = source_name.lower()
    if "clinical" in s:
        return f'<span class="badge badge-clinical">🧪 {source_name}</span>'
    elif "fda" in s:
        return f'<span class="badge badge-fda">🚨 {source_name}</span>'
    elif "pubmed" in s:
        return f'<span class="badge badge-pubmed">📚 {source_name}</span>'
    else:
        return f'<span class="badge badge-clinical">💊 {source_name}</span>'

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# USER CHAT INPUT
# -----------------------------------------------------------------------------
if prompt := st.chat_input("💬 Ask a medical query (e.g., 'What are side effects and clinical trials for UBT251?')..."):
    # Render user prompt
    st.chat_message("user").markdown(f"**Query:** {prompt}")
    st.session_state.messages.append({"role": "user", "content": f"**Query:** {prompt}"})

    with st.chat_message("assistant"):
        with st.spinner("🤖 Routing query & searching vector databases..."):
            try:
                # 1. Retrieve context using cached agentic retrieval
                retrieved_docs = cached_agentic_retrieve(prompt, top_k=TOP_K)
                
                # 2. Render Source Badges with Icons
                active_sources = [source for source, docs in retrieved_docs.items() if docs]
                if active_sources:
                    badges_html = " ".join([render_source_badge(src) for src in active_sources])
                    st.markdown(f"**🔍 Sources Evaluated:** {badges_html}", unsafe_allow_html=True)

                # 3. Generate structured answer via cached helper
                response_text = cached_generate_answer(prompt, retrieved_docs)
                
                # Render LLM output
                st.markdown(response_text)

                # 4. Display Collapsible Raw Evidence Section
                with st.expander("📄 View Retained Source Documents & Scores"):
                    for source, docs in retrieved_docs.items():
                        if docs:
                            st.markdown(f"### Source: {source}")
                            for idx, item in enumerate(docs, 1):
                                doc_text = item.get("document", "")
                                score = item.get("rerank_score", 0.0)
                                meta = item.get("metadata", {})
                                
                                st.markdown(
                                    f"""
                                    <div class="evidence-card">
                                        <strong>📍 Match #{idx}</strong> | <code>Relevance: {score:.3f}</code><br/>
                                        <small><strong>ID:</strong> {meta.get('id', 'N/A')}</small><br/>
                                        <p style="margin-top: 8px; margin-bottom: 0px;">{doc_text}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                # Append assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
