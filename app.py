import streamlit as st
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# Load the pretrained model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_documents_from_upload(uploaded_files):
    """Reads content from uploaded files."""
    documents = {}
    for uploaded_file in uploaded_files:
        documents[uploaded_file.name] = uploaded_file.read().decode("utf-8")
    return documents

def get_documents():
    """Reads all .txt files in the 'documents' directory."""
    documents = {}
    if not os.path.exists("documents"):
        os.makedirs("documents")
    for filename in os.listdir("documents"):
        if filename.endswith('.txt'):
            with open(os.path.join("documents", filename), 'r', encoding='utf-8', errors='ignore') as f:
                documents[filename] = f.read()
    return documents

def chunk_documents(documents):
    """Splits documents into chunks."""
    chunks = []
    for filename, content in documents.items():
        # Split on double newlines (paragraph separator)
        paragraphs = content.split('\\n\\n')
        for para in paragraphs:
            clean_para = para.strip().replace('\\n', ' ')
            if len(clean_para) > 30:  # Skip very short lines
                chunks.append({
                    "text": clean_para,
                    "source": filename
                })
    return chunks

def embed_chunks(chunks):
    """Generates embeddings for each chunk."""
    model = load_model()
    for chunk in chunks:
        chunk['embedding'] = model.encode(chunk['text'], convert_to_numpy=True)
    return chunks

def retrieve_top_chunks(query, chunks, top_k=3):
    """Retrieves the top k most relevant chunks."""
    model = load_model()
    # Step 1: Embed the query
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Step 2: Compute cosine similarity with each chunk
    similarities = []
    for chunk in chunks:
        sim = cosine_similarity(
            [query_embedding],
            [chunk['embedding']]
        )[0][0]
        similarities.append((chunk, sim))

    # Step 3: Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Return top K chunks
    top_chunks = similarities[:top_k]

    return top_chunks

def extract_metadata(text):
    """Extracts metadata from the resume text."""
    name = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)", text)
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\d{3}-\d{3}-\d{4}", text)
    return {
        "name": name.group(0) if name else "N/A",
        "email": email.group(0) if email else "N/A",
        "phone": phone.group(0) if phone else "N/A",
    }

def highlight_keywords(text, query):
    """Highlights keywords from the query in the text."""
    for word in query.split():
        text = re.sub(f"({word})", r"**\1**", text, flags=re.IGNORECASE)
    return text

def main():
    st.set_page_config(page_title="Resume Retriever", page_icon="📄", layout="wide")
    st.title("📄 Resume Retriever")
    st.write("This application helps you find the most relevant resumes based on your query.")

    st.sidebar.header("Upload Resumes")
    uploaded_files = st.sidebar.file_uploader("Upload your resume files (.txt)", accept_multiple_files=True, type="txt")

    documents = {}
    if uploaded_files:
        documents = get_documents_from_upload(uploaded_files)
    else:
        documents = get_documents()


    if not documents:
        st.warning("No documents found. Please upload some .txt files or add them to the 'documents' folder.")
        return

    st.sidebar.header("Uploaded Resumes")
    for filename in documents.keys():
        st.sidebar.markdown(f"- {filename}")


    query = st.text_input("🔍 Enter your query (e.g., 'software engineer with python experience'):", "")

    if query:
        with st.spinner("Chunking documents..."):
            chunks = chunk_documents(documents)
            time.sleep(1)
        st.success(f"Successfully chunked documents into {len(chunks)} paragraphs.")


        with st.spinner("Embedding chunks..."):
            chunks = embed_chunks(chunks)
            time.sleep(1)
        st.success("Successfully embedded all chunks.")


        with st.spinner("Retrieving top resumes..."):
            top_chunks = retrieve_top_chunks(query, chunks)
            time.sleep(1)
        st.header("📢 Top 3 Relevant Resumes:")
        for i, (chunk, score) in enumerate(top_chunks):
            metadata = extract_metadata(chunk['text'])
            st.subheader(f"🔹 Rank {i+1} | Score: {score:.4f} | Source: {chunk['source']}")
            st.write(f"**Name:** {metadata['name']} | **Email:** {metadata['email']} | **Phone:** {metadata['phone']}")
            highlighted_text = highlight_keywords(chunk['text'], query)
            st.write(highlighted_text)
            col1, col2 = st.columns(2)
            with col1:
                st.button("👍", key=f"up_{i}")
            with col2:
                st.button("👎", key=f"down_{i}")
            st.markdown("---")

if __name__ == "__main__":
    main()
