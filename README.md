# 📄 DocuMind: Secure Local OCR & Dynamic RAG System

DocuMind is a production-ready, 100% offline, and fully localized document processing pipeline. It features an advanced, searchable Retrieval-Augmented Generation (RAG) system capable of handling multilingual and bilingual datasets (Bangla and English) safely on your local hardware without relying on third-party commercial cloud APIs.

---

## 🚀 Key Features
- **Local Multilingual OCR:** Extract text from handwritten, digital, or complex scanned images locally.
- **Dynamic Hybrid Search:** Combine strict manual metadata filtering (Language, Type, Date) with vector similarity calculations.
- **Privacy-First RAG Pipeline:** Secure offline execution utilizing localized open-source embeddings and LLMs.

---

## 🏗️ System Architecture & Technical Specifications



### 1. Local OCR Model Selection & Trade-offs
- **Model Choice:** **EasyOCR** (underpinned by ResNet for feature extraction and LSTM for sequence labeling).
- **Trade-offs & Performance:** Traditional OCR engines like Tesseract often suffer severe baseline accuracy degradation when handling complex, non-linear **Bangla scripts** (such as joint-letters/যুক্তাক্ষর and complex inflections). EasyOCR uses deep learning-based spatial transformers which drastically improve text alignment and baseline extraction performance for bilingual Bengali-English layout sheets, shifting the compute bottleneck slightly to CPU/GPU execution without breaking the localized compliance rule.

### 2. Text-Chunking & Embedding Strategy
- **Chunking Mechanism:** **RecursiveCharacterTextSplitter** from LangChain. 
- **Parameters:** `chunk_size=500` characters, `chunk_overlap=50` characters.
- **Context Preservation:** This configuration guarantees that paragraph-level semantic transitions and metadata contexts are not severely truncated in bilingual layouts, ensuring clean token windows for inference.
- **Embedding Vector Space:** **nomic-embed-text** mapped inside **ChromaDB**. It naturally supports cross-lingual text embeddings, creating accurate semantic vectors across both English and Bengali text blocks.

### 3. Dynamic Hybrid Filtering Engine
The search process operates as a pipeline:
1. **Hard Constraint Filtering:** When a user queries the database, ChromaDB evaluates the explicit configuration filters first via relational operators (`$and` logical blocks with strict property checking).
2. **Soft Semantic Search:** Only the vector documents passing the initial metadata layer are subjected to **Cosine Similarity** mapping.
3. **Inference Context Generation:** This limits the token scope injected into **Llama 3.2**, eliminating hallucinations stemming from mismatched document profiles.

---

## 🛠️ Installation & Setup Guidelines

### Prerequisites
- Python 3.10 to 3.14
- Ollama runtime installed locally

### Step 1: Pull Local Models via Ollama
Run the following commands in your terminal to cache the embedding and inference weights locally:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2



---
### Step 2: Clone & Install Dependencies
```bash
git clone [https://github.com/YOUR_USERNAME/local-ocr-dynamic-rag.git](https://github.com/YOUR_USERNAME/local-ocr-dynamic-rag.git)
cd local-ocr-dynamic-rag

# Install requirements
pip install -r requirements.txt
(Note: Create a requirements.txt featuring: streamlit, easyocr, langchain, langchain-community, langchain-text-splitters, chromadb, pillow, numpy)

Step 3: Run the Local Application
```bash
streamlit run app.py

📊 Database Schema (ChromaDB Vector Metadata)
Each injected document chunk is tightly coupled with the following schema mappings:

JSON
{
  "id": "UUID-String",
  "document_chunk": "Extracted text payload goes here...",
  "metadata": {
    "language": "bangla | english",
    "type": "Invoice | Agreement | Report | Other",
    "date": "YYYY-MM-DD",
    "source": "filename.png"
  }
}
🎥 Project Validation Demo
Demo Presentation Video Link: [Insert Your Recorded Demo Video Link Here]

