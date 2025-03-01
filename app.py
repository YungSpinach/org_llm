import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from docx import Document

# Load documents
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append(text)
        elif filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            documents.append(text)
        elif filename.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents.append(text)
    return documents

# Process documents
def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode([chunk.page_content for chunk in chunks])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return chunks, index, embedding_model

# Load LLM
def load_llm(model_name="deepseek-ai/deepseek-llm"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Semantic search
def search(query, index, chunks, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i].page_content for i in indices[0]]

# Generate response
def generate_response(query, context, tokenizer, model):
    input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit App
st.title("LLM Document Querying App")

# Preload documents
folder_path = "C:\Users\chris\OneDrive\Documents\GitHub\org_llm\documents"  # Replaced with document folder
documents = load_documents(folder_path)
chunks, index, embedding_model = process_documents(documents)

# Load LLM
tokenizer, model = load_llm()

# Query interface
query = st.text_input("Enter your query:")
if query:
    relevant_chunks = search(query, index, chunks, embedding_model)
    context = "\n".join(relevant_chunks)
    response = generate_response(query, context, tokenizer, model)
    st.write("Response:", response)