import os
import sys
import hashlib
from io import BytesIO

# --- Fix Windows console encoding ---
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# --- LangChain imports ---
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- OCR imports ---
from pdf2image import convert_from_path
import pytesseract

# ✅ Set paths manually (update if your installation differs)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPLER_PATH = r"C:\poppler-25.07.0\Library\bin"

# --- Config ---
pdf_folder = "./pdfs"
db_location = "./chroma_langchain_db"
collection_name = "college_policies"

os.makedirs(pdf_folder, exist_ok=True)
os.makedirs(db_location, exist_ok=True)

def file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# --- Initialize embeddings and Chroma ---
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=db_location,
    embedding_function=embeddings
)

# --- Load existing metadata ---
existing_docs = vector_store.get(include=["metadatas"])
existing_hashes = set()
metadatas = existing_docs.get("metadatas", []) if existing_docs else []
for meta in metadatas:
    if meta and "file_hash" in meta:
        existing_hashes.add(meta["file_hash"])

# --- Process PDFs ---
all_documents = []
print("\n🔍 Scanning PDF folder for new or updated files...\n")

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        fhash = file_hash(pdf_path)

        if fhash not in existing_hashes:
            print(f"📄 New or updated file detected: {filename}")

            try:
                loader = PDFPlumberLoader(pdf_path)
                pages = loader.load()
                print(f"   → Loaded {len(pages)} pages")

                # Combine text
                full_text = " ".join([p.page_content.strip() for p in pages if p.page_content.strip()])

                # --- OCR fallback if needed ---
                if len(full_text.strip()) < 50:
                    print(f"⚠️  No readable text found in {filename}, using OCR...")
                    try:
                        images = convert_from_path(pdf_path, poppler_path=POPLER_PATH)
                    except Exception as e:
                        print(f"❌ Poppler error: {e}")
                        continue

                    ocr_text = ""
                    for i, img in enumerate(images, start=1):
                        text = pytesseract.image_to_string(img)
                        if text.strip():
                            ocr_text += f"\n--- Page {i} ---\n{text.strip()}"

                    if not ocr_text.strip():
                        print(f"❌ OCR also failed for {filename}, skipping.")
                        continue

                    pages = [Document(page_content=ocr_text, metadata={"source": filename})]
                    print(f"   ✅ OCR extracted {len(ocr_text.split())} words from {filename}")

                # Split into smaller chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                docs = splitter.split_documents(pages)

                for d in docs:
                    d.metadata["source"] = filename
                    d.metadata["file_hash"] = fhash

                all_documents.extend(docs)
                print(f"   ✅ Prepared {len(docs)} chunks from {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

        else:
            print(f"✅ Skipping already indexed file: {filename}")

# --- Add to Chroma ---
if all_documents:
    vector_store.add_documents(all_documents)
    print(f"\n✅ Added {len(all_documents)} new chunks to Chroma.")
else:
    print("\n📦 No new PDFs to embed. Using existing database.")

# --- Create retriever ---
retriever = vector_store.as_retriever(search_kwargs={"k": 15})

try:
    total_docs = len(vector_store.get(include=["documents"])["documents"])
    print(f"\n📚 Total embedded text chunks in database: {total_docs}")
except Exception as e:
    print(f"⚠️ Could not count documents: {e}")
