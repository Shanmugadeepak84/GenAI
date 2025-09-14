import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

# LangChain imports
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
    from langchain.docstore.document import Document
except Exception as e:
    st.error(f"Error importing langchain modules: {e}")
    raise

UPLOADS_DIR = Path("./uploads")
VECTORSTORE_DIR = Path("./vectorstore")
UPLOADS_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

def load_documents_from_files(uploaded_files) -> List[Document]:
    docs = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        tmp.close()
        path = tmp.name
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif suffix in [".txt"]:
                loader = TextLoader(path, encoding='utf-8')
                docs.extend(loader.load())
            elif suffix in [".docx", ".doc"]:
                try:
                    loader = UnstructuredWordDocumentLoader(path)
                    docs.extend(loader.load())
                except Exception:
                    st.warning(f"Could not parse Word file {uploaded_file.name}. Skipping.")
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))
        except Exception as e:
            st.warning(f"Failed to load {uploaded_file.name}: {e}")
    return docs

def load_local_folder(folder_path: str) -> List[Document]:
    p = Path(folder_path)
    docs = []
    if not p.exists():
        return docs
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in [".pdf", ".txt", ".docx", ".doc"]:
            try:
                if f.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(f))
                    docs.extend(loader.load())
                elif f.suffix.lower() == ".txt":
                    loader = TextLoader(str(f), encoding='utf-8')
                    docs.extend(loader.load())
                else:
                    try:
                        loader = UnstructuredWordDocumentLoader(str(f))
                        docs.extend(loader.load())
                    except Exception:
                        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                            docs.append(Document(page_content=fh.read(), metadata={"source": str(f)}))
            except Exception as e:
                st.warning(f"Error loading {f}: {e}")
    return docs

def create_vectorstore(docs: List[Document], persist_dir: Path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(str(persist_dir))
    return vectordb

def load_vectorstore(persist_dir: Path):
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        return None
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(persist_dir), embeddings)

def make_qa_chain(vectordb):
    llm = OpenAI(temperature=0)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

st.set_page_config(page_title="Streamlit RAG (LangChain + OpenAI)", layout="wide")
st.title("🧠 Streamlit RAG — Talk to your documents")

st.sidebar.header("Ingest / Vectorstore")
api_key = st.sidebar.text_input("OpenAI API key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

uploaded_files = st.sidebar.file_uploader("Upload files (pdf, txt, docx)", accept_multiple_files=True)
use_local_folder = st.sidebar.checkbox("Load files from local folder (./uploads)")
local_folder_path = st.sidebar.text_input("Local folder (relative)", value=str(UPLOADS_DIR))

if uploaded_files:
    st.sidebar.write(f"Uploaded {len(uploaded_files)} files")

if st.sidebar.button("Ingest uploaded files"):
    if not uploaded_files:
        st.sidebar.warning("No files uploaded — choose files first")
    else:
        with st.spinner("Loading documents and creating vectorstore — this can take a minute..."):
            docs = load_documents_from_files(uploaded_files)
            if docs:
                vectordb = create_vectorstore(docs, VECTORSTORE_DIR)
                st.sidebar.success("Vectorstore created and persisted to ./vectorstore")
            else:
                st.sidebar.error("No documents were loaded — check file types")

if use_local_folder and st.sidebar.button("Ingest local folder"):
    with st.spinner("Loading local folder and creating vectorstore..."):
        docs = load_local_folder(local_folder_path)
        if docs:
            vectordb = create_vectorstore(docs, VECTORSTORE_DIR)
            st.sidebar.success("Vectorstore created and persisted to ./vectorstore")
        else:
            st.sidebar.error("No documents found in folder")

vectordb = None
if st.sidebar.button("Load persisted vectorstore"):
    with st.spinner("Loading vectorstore..."):
        vectordb = load_vectorstore(VECTORSTORE_DIR)
        if vectordb:
            st.sidebar.success("Vectorstore loaded")
        else:
            st.sidebar.error("No persisted vectorstore found. Ingest first.")

st.header("Ask questions — retrieval-augmented answers")
if not vectordb:
    st.info("No vectorstore loaded. Ingest documents (sidebar) or load persisted vectorstore.")

query = st.text_input("Ask a question about your documents:")
if st.button("Run") and query:
    if not vectordb:
        st.error("No vectorstore loaded. Please ingest or load a vectorstore first.")
    else:
        with st.spinner("Running retrieval + LLM..."):
            qa = make_qa_chain(vectordb)
            res = qa(query)
            answer = res.get("result") if isinstance(res, dict) else res
            st.subheader("Answer")
            st.write(answer)
            docs = res.get("source_documents") if isinstance(res, dict) else None
            if docs:
                st.subheader("Source documents / chunks")
                for i, d in enumerate(docs):
                    st.markdown(f"**Source {i+1} — {d.metadata.get('source','unknown')}**")
                    st.write(d.page_content[:1000])

st.markdown("---")
st.caption("This app uses OpenAI for embeddings & completion via LangChain, and FAISS for vector storage.")
