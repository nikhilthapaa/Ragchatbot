import streamlit as st
import os
import tempfile
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- DIRECTly SETTING YOUR API KEYS ---
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-GhtWcC6M9YuIZCzjmVwsW5vlw1iQTrCSjQb9QBuyK22yet57"
os.environ["GROQ_API_KEY"] = "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"

# 1. PAGE CONFIG & THEME INITIALIZATION
st.set_page_config(page_title="LlamaParse + Groq Enterprise RAG", layout="wide", initial_sidebar_state="expanded")

# Session States for Authentication and Theme
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "parsed_docs" not in st.session_state:
    st.session_state.parsed_docs = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# --- CUSTOM CSS FOR MODERN DARK/LIGHT MODE ---
def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
            <style>
            .stApp { background-color: #0E1117; color: #FFFFFF; }
            .stTabs [data-baseweb="tab-list"] { gap: 10px; }
            .stTabs [data-baseweb="tab"] { background-color: #1E293B; border-radius: 4px 4px 0px 0px; padding: 10px 20px; color: white; }
            .stTabs [aria-selected="true"] { background-color: #3B82F6 !important; }
            div[data-testid="stExpander"] { background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; }
            .chunk-box { background-color: #1E293B; padding: 15px; border-radius: 8px; border-left: 5px solid #10B981; margin-bottom: 10px; color: white; }
            .retrieved-box { background-color: #1E3A8A; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6; margin-bottom: 10px; color: white; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #F8FAFC; color: #0F172A; }
            .stTabs [data-baseweb="tab-list"] { gap: 10px; }
            .stTabs [data-baseweb="tab"] { background-color: #E2E8F0; border-radius: 4px 4px 0px 0px; padding: 10px 20px; color: #0F172A; }
            .stTabs [aria-selected="true"] { background-color: #3B82F6 !important; color: white !important; }
            div[data-testid="stExpander"] { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; }
            .chunk-box { background-color: #FFFFFF; padding: 15px; border-radius: 8px; border-left: 5px solid #10B981; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; margin-bottom: 10px; color: #334155; }
            .retrieved-box { background-color: #EFF6FF; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6; margin-bottom: 10px; color: #1E3A8A; }
            </style>
        """, unsafe_allow_html=True)

apply_theme()

# --- INTERFACE 1: LOGIN PAGE ---
if not st.session_state.logged_in:
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown("<h2 style='text-align: center;'>🔐 RAG Chatbot Portal</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password", value="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")
    st.stop()

# --- INTERFACE 2: MAIN DASHBOARD (LOGGED IN) ---

# Top Header Bar
header_col1, header_col2 = st.columns([8, 2])
with header_col1:
    st.title("🦙 LlamaParse + Groq Multi-PDF RAG")
with header_col2:
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        if st.button("🌓 Theme"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    with t_col2:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.parsed_docs = None
            st.session_state.chunks = None
            st.session_state.query_engine = None
            st.rerun()

st.markdown("---")

# Global LLM and Embedding Configuration using LlamaIndex Settings
Settings.llm = Groq(model="llama3-70b-8192", api_key=os.environ.get("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Control Panel")
uploaded_files = st.sidebar.file_uploader("Upload Multiple PDFs", type=["pdf"], accept_multiple_files=True)

st.sidebar.markdown("### 🧩 Chunk Configurations")
chunk_size = st.sidebar.slider("Chunk Size", min_value=128, max_value=1024, value=512, step=64)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=128, value=50, step=10)

if st.sidebar.button("🚀 Parse & Build Pipeline", use_container_width=True) and uploaded_files:
    with st.spinner("Processing through LlamaParse..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # LlamaParse setup
            parser = LlamaParse(result_type="markdown")
            file_extractor = {".pdf": parser}
            
            reader = SimpleDirectoryReader(input_dir=temp_dir, file_extractor=file_extractor)
            docs = reader.load_data()
            st.session_state.parsed_docs = docs

    with st.spinner("Splitting text into chunks & generating embeddings..."):
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(st.session_state.parsed_docs)
        st.session_state.chunks = nodes

    with st.spinner("Indexing into Vector Database..."):
        index = VectorStoreIndex(nodes)
        st.session_state.query_engine = index.as_query_engine(similarity_top_k=3)
        st.sidebar.success(f"✅ Successfully setup {len(uploaded_files)} files!")

# --- MAIN TAB INTERACTION ---
tab1, tab2, tab3 = st.tabs(["📄 Parsed Markdown View", "🧩 Chunk Inspector", "💬 Multi-PDF Chatbot (Groq)"])

# TAB 1: Parsed Documents Visualizer
with tab1:
    st.subheader("📋 Document Content After LlamaParse")
    if st.session_state.parsed_docs:
        for idx, doc in enumerate(st.session_state.parsed_docs):
            file_name = doc.metadata.get("file_name", f"Document {idx+1}")
            with st.expander(f"📁 Source: {file_name}"):
                st.markdown(doc.text)
    else:
        st.info("Sidebar bata PDF upload garera process garnuhos.")

# TAB 2: Chunk Inspector
with tab2:
    st.subheader("✂️ Document Segments (Chunks)")
    if st.session_state.chunks:
        st.write(f"**Total Chunks Created:** {len(st.session_state.chunks)}")
        for idx, node in enumerate(st.session_state.chunks):
            file_name = node.metadata.get("file_name", "Unknown Document")
            st.markdown(f"""
            <div class='chunk-box'>
                <strong>Chunk {idx+1}</strong> | Source: <i>{file_name}</i><br><br>
                {node.text}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Chunks haru herna paila document parse garna parcha.")

# TAB 3: Q&A Chatbot with Source Tracer
with tab3:
    st.subheader("💬 Query Engine powered by Groq")
    if st.session_state.query_engine:
        query_str = st.text_input("Ask anything from any uploaded PDFs:")
        if query_str:
            with st.spinner("Searching document pipeline & generating response via Groq..."):
                response = st.session_state.query_engine.query(query_str)
                
                # Render System Answer
                st.markdown("### 💡 System Response:")
                st.info(response.response)
                
                st.markdown("---")
                
                # Render Retrieved Nodes (RAG Tracer)
                st.markdown("### 🎯 Nodes Retrieved (RAG Pipeline Proof):")
                for idx, source_node in enumerate(response.source_nodes):
                    f_name = source_node.node.metadata.get('file_name', 'Unknown')
                    score = source_node.score
                    st.markdown(f"""
                    <div class='retrieved-box'>
                        <strong>Retrieved Source Node {idx+1}</strong> (Cosine Similarity Score: <code style='color:#FF4B4B;'>{score:.4f}</code>)<br>
                        <strong>File Source:</strong> {f_name}<br><br>
                        {source_node.node.text}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Chatbot use garna agadi kindly files parse garnuhos.")