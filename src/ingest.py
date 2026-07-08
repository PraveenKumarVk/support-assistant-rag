import uuid
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pypdf
import pytesseract
from pdf2image import convert_from_path
from bs4 import BeautifulSoup, NavigableString, Tag
import argparse
from . import config
from pathlib import Path
from rank_bm25 import BM25Okapi
import re
import pickle

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))


def chunk_text(text:str, chunk_size:int, overlap:int):
    chunks = []
    if overlap >= chunk_size:
        raise ValueError(f"chunk_size ({chunk_size}) must be more than overlap({overlap})")
    for i in range(0, len(text),chunk_size-overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def load_markdown_file(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        raise FileNotFoundError(f"No file exists at: {file_path}")
        

def chunk_markdown_file(file_path:str, chunk_size:int, overlap:int):
    file_content = load_markdown_file(file_path)
    chunks = chunk_text(file_content, chunk_size, overlap)
    file_name = os.path.basename(file_path)
    chunk_store = {}
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        chunk_store[chunk_id] = {'chunk_text':chunk, 'file_path':file_path, 'file_type':"markdown", 'location': file_name}
    return chunk_store


def embed_chunks(chunk_store: dict):
    embed_values = embed_model.encode([value['chunk_text'] for _, value in chunk_store.items()])
    for (key,_), value in zip(chunk_store.items(), embed_values):
        chunk_store[key]['embedding'] = value
    return chunk_store
        
def store_chunks(chunk_store):
    index_name = 'support-assistant-rag'
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type='dense',
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    vectors = []
    for chunk_id, chunk in chunk_store.items():
        vectors.append({'id':chunk_id, 'values': [float(x) for x in chunk['embedding']], 'metadata':{k:v for k,v in chunk.items() if k != 'embedding'}})

    index = pc.Index(index_name)
    index.upsert(vectors=vectors, namespace='md-vectors')
    

def detect_pdf_type(file_path:str):
    text_len = 0
    pages = 0
    with open(file_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_len += len(text)
            pages+=1
            
    return "text_native" if text_len>0 and pages>0 and text_len//pages > 70 else "scanned"

def load_text_native_pdf(file_path: str):
    text = []
    with open(file_path,"rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def load_scanned_pdf(file_path: str):
    pages = convert_from_path(file_path,dpi=300)
    texts = []
    for page in pages:
        page_text = pytesseract.image_to_string(page, lang='eng')
        texts.append(page_text or "")
    return "\n".join(texts)
    
def load_pdf(file_path):
    pdf_type = detect_pdf_type(file_path=file_path)
    text = load_text_native_pdf(file_path) if pdf_type == "text_native" else load_scanned_pdf(file_path)
    return text

def chunk_pdf_file(file_path:str, chunk_size:int, overlap:int):
    text = load_pdf(file_path)
    chunks = chunk_text(text, chunk_size, overlap)
    chunk_store = {}
    file_name = os.path.basename(file_path)
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        chunk_store[chunk_id] = {'chunk_text':chunk, 'file_path':file_path, 'file_type':"pdf", 'location': file_name}
    return chunk_store


def load_html_file(file_path:str):
    try:
        with open(file_path,"r",encoding='utf-8') as file:
            html_content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"No file exists at: {file_path}")

    soup = BeautifulSoup(html_content,"html.parser")
    for boilerplate in soup.find_all(["script", "style", "noscript", "nav", "header", "footer"]):
        boilerplate.decompose()

    headings = soup.find_all(["h1","h2","h3","h4","h5","h6"])
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    structured_sections = []

    if not headings:
        scope = soup.body or soup
        body_text = scope.get_text(separator=" ", strip=True)
        if body_text:
            structured_sections.append(("", body_text))
        return structured_sections

    for i, heading in enumerate(headings):
        heading_text = heading.get_text(separator=" ", strip=True)
        if not heading_text:
            continue
            
        content_pieces = []
        seen = set()
        
        next_heading = headings[i + 1] if i + 1 < len(headings) else None
        
        current = heading.next_element
        while current and current != next_heading:
            if isinstance(current, Tag) and current.name in heading_tags:
                break
                
            if isinstance(current, NavigableString):
                text_content = current.strip()
                if text_content and heading not in current.parents and text_content not in seen:
                    content_pieces.append(text_content)
                    seen.add(text_content)
                    
            current = current.next_element
            
        content_text = " ".join(content_pieces)
        
        if content_text.strip():
            structured_sections.append((heading_text, content_text))
            
    return structured_sections

def chunk_html_file(file_path:str, chunk_size:int, overlap:int):
    structured_sections = load_html_file(file_path)
    chunk_store = {}
    for section in structured_sections:
        chunks = chunk_text(section[1], chunk_size, overlap)
        location = section[0] or ""
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_store[chunk_id] = {'chunk_text':chunk, 'file_path':file_path, 'file_type':"html", 'location': location}
    return chunk_store

def build_bm25_index(chunk_store:dict):
    documents = []
    chunk_ids = []
    for chunk_id, chunk in chunk_store.items():
        documents.append(chunk['chunk_text'])
        chunk_ids.append(chunk_id)
    tokenized_corpus = [re.sub(r'[^\w\s-]', '', doc).lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump((bm25, chunk_ids), f)
    return bm25, chunk_ids

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default=str(config.CORPUS_DIR))
    args = parser.parse_args()
    corpus_dir = Path(args.corpus).resolve()
    chunk_store = {}
    
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".md"):
                    chunk_store.update(chunk_markdown_file(file_path,1000,200))
                elif file.endswith(".html"):
                    chunk_store.update(chunk_html_file(file_path,1000,200))
                elif file.endswith(".pdf"):
                    chunk_store.update(chunk_pdf_file(file_path,1000,200))
            except Exception as ex:
                print(f"Error loading file: {ex}")
    build_bm25_index(chunk_store)
    embedded_chunk_store = embed_chunks(chunk_store)
    store_chunks(embedded_chunk_store)
    
if __name__=="__main__":
    main()