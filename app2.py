import os
import streamlit as st
from dotenv import load_dotenv
from llama_parse import LlamaParse
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from llama_index.llms.groq import Groq

# Load environment variables from .env file securely in the background
load_dotenv()

LLAMA_API_KEY = os.environ.get("LLAMA_Parse_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Advanced RAG Inspector", page_icon="🤖", layout="wide")

# Custom CSS for a beautiful, modern UI
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 20px; }
    .section-header { font-size: 1.5rem; font-weight: 600; margin-top: 20px; margin-bottom: 10px; }
    .card { padding: 15px; border-radius: 10px; background-color: #f8f9fa; border: 1px solid #e0e0e0; margin-bottom: 15px; }
    .dark .card { background-color: #262730; border: 1px solid #464646; }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States
if "parsed_docs" not in st.session_state:
    st.session_state.parsed_docs = {}  # {filename: raw_text}
if "chunks" not in st.session_state:
    st.session_state.chunks = []  # List of dicts: {"text": str, "source": str}
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_retrieval" not in st.session_state:
    st.session_state.latest_retrieval = []

# Load Embedding Model (Cached to load only once)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# --- 2. HELPER FUNCTIONS ---
def chunk_text(text, source_name, chunk_size=800, chunk_overlap=100):
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({"text": chunk_text, "source": source_name})
        if i + chunk_size >= len(words):
            break
    return chunks

def build_vector_store(chunks):
    """Encodes chunks and loads them into a FAISS index."""
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve_chunks(query, index, chunks, k=3):
    """Retrieves top K chunks matching the query."""
    query_vector = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    
    retrieved = []
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved.append(chunks[idx])
    return retrieved

def generate_llm_response(query, retrieved_contexts, groq_api_key):
    """Generates LLM response using Groq with retrieved context."""
    greetings = ["hello", "hi", "hey", "namaste", "good morning", "good afternoon", "wasup"]
    if any(g == query.lower().strip().split()[0] for g in greetings) or len(query.strip()) < 4:
        return "Hello! 👋 I am your interactive AI assistant. I can help you analyze your uploaded documents. Ask me anything about them!"
        
    if not retrieved_contexts:
        return "I couldn't find any relevant context to answer that from your files."

    try:
        context_str = "\n---\n".join([f"Source ({c['source']}): {c['text']}" for c in retrieved_contexts])
        
        system_prompt = (
            "You are a helpful assistant. Use the following pieces of context to answer the user's question. "
            "If you don't know the answer or if it's not in the context, say that you don't know based on the documents. "
            "Do not make things up.\n\n"
            f"Context:\n{context_str}"
        )
        
        # FIXED: Updated deprecated llama3-8b-8192 to the active llama-3.1-8b-instant endpoint
        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)
        response = llm.complete(f"{system_prompt}\n\nUser Question: {query}")
        return str(response)
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"

# --- 3. SIDEBAR: UPLOADS ONLY ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 Parse & Process", use_container_width=True):
        if not LLAMA_API_KEY:
            st.error("LlamaParse API Key missing in your .env file!")
        elif not uploaded_files:
            st.error("Please upload at least one PDF file!")
        else:
            with st.spinner("Parsing with LlamaParse & Building Index..."):
                try:
                    parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="text")
                    
                    new_chunks = []
                    for uploaded_file in uploaded_files:
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        parsed_data = parser.load_data(uploaded_file.name)
                        file_text = "\n".join([doc.text for doc in parsed_data])
                        
                        st.session_state.parsed_docs[uploaded_file.name] = file_text
                        
                        file_chunks = chunk_text(file_text, uploaded_file.name)
                        new_chunks.extend(file_chunks)
                        
                        os.remove(uploaded_file.name)
                    
                    st.session_state.chunks = new_chunks
                    st.session_state.vector_index = build_vector_store(new_chunks)
                    st.success("Processing complete! Pipeline loaded successfully.")
                except Exception as e:
                    st.error(f"Error during parsing pipeline processing: {str(e)}")

# --- 4. MAIN INTERFACE ---
st.markdown('<div class="main-header">🤖 Transparent RAG Chatbot</div>', unsafe_allow_html=True)

# Tabs for Pipeline Inspection
tab_chat, tab_parsed, tab_chunks, tab_retrieval = st.tabs([
    "💬 Chat Assistant", 
    "📄 Step 1: Parsed Text View", 
    "✂️ Step 2: Chunk Inspector", 
    "🔍 Step 3: Retrieval Monitor"
])

# --- TAB 1: CHAT INTERFACE ---
with tab_chat:
    st.markdown('<div class="section-header">Ask Anything About Your Documents</div>', unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if user_query := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        greetings = ["hello", "hi", "hey", "namaste", "wasup", "yo"]
        is_greeting = any(g == user_query.lower().strip().split()[0] for g in greetings)
        
        if is_greeting:
            response = "Hello! 👋 How can I help you today? Ask me any questions regarding your uploaded PDFs."
            retrieved = []
        elif st.session_state.vector_index is not None and st.session_state.chunks:
            if not GROQ_API_KEY:
                response = "⚠️ Groq API Key is missing in your background configuration environment!"
                retrieved = []
            else:
                retrieved = retrieve_chunks(user_query, st.session_state.vector_index, st.session_state.chunks)
                st.session_state.latest_retrieval = retrieved
                response = generate_llm_response(user_query, retrieved, GROQ_API_KEY)
        else:
            response = "I'd love to help, but please upload and process your PDFs first using the sidebar! 📂"
            retrieved = []
            
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- TAB 2: PARSED TEXT VIEW ---
with tab_parsed:
    st.markdown('<div class="section-header">Raw LlamaParse Outputs</div>', unsafe_allow_html=True)
    if st.session_state.parsed_docs:
        selected_doc = st.selectbox("Select document to inspect:", list(st.session_state.parsed_docs.keys()))
        st.text_area(f"Full Extracted Text for {selected_doc}", st.session_state.parsed_docs[selected_doc], height=400)
    else:
        st.info("No documents parsed yet. Use the sidebar to upload files.")

# --- TAB 3: CHUNK INSPECTOR ---
with tab_chunks:
    st.markdown('<div class="section-header">Document Split Blocks (Chunks)</div>', unsafe_allow_html=True)
    if st.session_state.chunks:
        st.write(f"Total Chunks Generated: **{len(st.session_state.chunks)}**")
        for idx, chunk in enumerate(st.session_state.chunks[:15]): 
            with st.expander(f"🧩 Chunk {idx+1} | Source: {chunk['source']}"):
                st.code(chunk['text'], language="text")
        if len(st.session_state.chunks) > 15:
            st.caption("Showing first 15 chunks for crisp rendering performance.")
    else:
        st.info("Chunks will be displayed here once your documents are processed.")

# --- TAB 4: RETRIEVAL MONITOR ---
with tab_retrieval:
    st.markdown('<div class="section-header">Vector Database (FAISS) Matches</div>', unsafe_allow_html=True)
    if st.session_state.latest_retrieval:
        st.write("These are the most relevant document pieces sent to the LLM for your last question:")
        for rank, chunk in enumerate(st.session_state.latest_retrieval):
            st.markdown(f"""
            <div class="card">
                <b>🎯 Rank {rank+1} Match</b> | <small>Source: {chunk['source']}</small><br/>
                <p style="margin-top:10px; font-style: italic;">"{chunk['text']}"</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Ask a question in the chat tab to see real-time vector retrieval analytics here.")