# 🧠 Local AI Agent with RAG (Retrieval-Augmented Generation)

Watch the demo video here:  
👉 Demo Video(https://drive.google.com/file/d/1VhIiqNOfoHOUOEbZI8nlsGpyS93-IJoE/view?usp=sharing)


This project is a **local AI chatbot** powered by **Streamlit**, **LangChain**, **Ollama**, and **Chroma**.  
It retrieves answers from embedded PDF documents using a Retrieval-Augmented Generation (RAG) pipeline.  

---

⚙️ 1. Setup Virtual Environment

### 🧩 Create a Virtual Environment

`python -m venv venv`


### 📦 2. Install Dependencies

Make sure your virtual environment is activated, then install all required libraries:
`pip install -r requirements.txt`


### 🧰 3. Install External Tools
🟣 Tesseract OCR

Used for text extraction from scanned PDFs.
🔗 Download: Tesseract OCR (UB Mannheim)
https://github.com/UB-Mannheim/tesseract/wiki

After installation:

Ensure the path (e.g. C:\Program Files\Tesseract-OCR) is added to your system environment variables.

🔵 Poppler

Used by pdf2image to convert PDF pages to images for OCR.
🔗 Download: Poppler for Windows
https://github.com/oschwartz10612/poppler-windows/releases

After downloading:

Extract the ZIP file (e.g., poppler-25.07.0) to C:\.

Add the bin path to your system environment variables.
Example: C:\poppler-25.07.0\Library\bin



### 🤖 4. Ollama Setup

This project uses Ollama for local LLM inference.

🪶 Install Ollama

Download from: https://ollama.ai/download

Then, pull the required models:

`ollama pull llama3.2`
`ollama pull mxbai-embed-large`


Chat model: llama3.2

Embedding model: mxbai-embed-large

### 🧩 5. Run the Application

After setup, run the chatbot locally with:

`streamlit run app.py`
Once it starts, open the provided localhost URL (usually http://localhost:8501) in your browser.
