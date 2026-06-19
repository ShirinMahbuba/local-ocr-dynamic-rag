import streamlit as st
import easyocr
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import numpy as np

# ১. প্রাথমিক সেটআপ ও মডেল লোড
st.set_page_config(page_title="DocuMind: Local OCR & RAG", layout="wide")
st.title("📄 DocuMind: Secure Local OCR & Dynamic RAG System")

@st.cache_resource
def load_ocr():
    # বাংলা ও ইংরেজি দুই ভাষার জন্যই OCR রিডার লোড করা হচ্ছে
    return easyocr.Reader(['bn', 'en'], gpu=False)

reader = load_ocr()
# 'llama3' এর পরিবর্তে আপনার ডাউনলোড করা 'llama3.2' মডেলটি এখানে লিখে দিন
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = Ollama(model="llama3.2")

# ২. লোকাল ডাটাবেস ডিরেক্টরি সেটআপ
DB_DIR = "./chroma_db"
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)


def build_chroma_filter(**fields):
    """
    Chroma-র where filter-এ একসাথে একাধিক field দিলে অবশ্যই প্রতিটাকে
    operator ($eq) দিয়ে wrap করতে হয়, আর ২+ field হলে $and দিয়ে combine
    করতে হয় — নাহলে ValueError("Expected where to have exactly one
    operator...") আসে। এই হেল্পার সেটাই সঠিকভাবে বানিয়ে দেয়, আর কোনো
    field None/খালি থাকলে সেটা বাদ দিয়ে দেয় (যাতে filter optional রাখা যায়)।
    """
    conditions = [{key: {"$eq": value}} for key, value in fields.items() if value]
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# সাইডবার - ম্যানুয়াল মেটাডেটা ফিল্টারিংয়ের জন্য ইনপুট
st.sidebar.header("🎛️ Manual Metadata Filters")
doc_lang = st.sidebar.selectbox("Document Language", ["bangla", "english"])
doc_type = st.sidebar.selectbox("Document Type", ["Invoice", "Agreement", "Report", "Other"])
doc_date = st.sidebar.date_input("Document Date")

# ৩. ফাইল আপলোড সেকশন
uploaded_file = st.file_uploader("Upload a Scanned Image (Bangla/English)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document", width=400)
    
    if st.button("🚀 Process & Extract Text Locally"):
        with st.spinner("Extracting text using Local EasyOCR..."):
            # ইমেজকে নুমি অ্যারেতে কনভার্ট করে OCR করা
            image_np = np.array(image)
            ocr_results = reader.readtext(image_np, detail=0)
            full_text = "\n".join(ocr_results)
            
            st.success("Extraction Complete!")
            st.subheader("📝 Extracted Raw Text:")
            st.text_area("", full_text, height=150)
            
            # ৪. টেক্সট চ্যাংকিং ও ভেক্টর ডাটাবেসে সেভ করা
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(full_text)
            
            # মেটাডেটা তৈরি করা
            metadata = {
                "language": doc_lang,
                "type": doc_type,
                "date": str(doc_date),
                "source": uploaded_file.name
            }
            metadatas = [metadata] * len(chunks)
            
            # ChromaDB-তে চ্যাংক ও মেটাডেটা সেভ
            vector_store = Chroma.from_texts(
                texts=chunks, 
                embedding=embeddings, 
                metadatas=metadatas, 
                persist_directory=DB_DIR
            )
            st.info(f"Successfully chunked into {len(chunks)} pieces and stored inside Local Vector DB with metadata!")

# 🔍 ৫. ডাইনামিক র‌্যাগ (RAG) সার্চ ও চ্যাট সেকশন
st.write("---")
st.subheader("💬 Ask Questions based on Filtered Documents")
user_query = st.text_input("Enter your natural language query here:")

if user_query:
    # ডাটাবেস কানেক্ট করা
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # মেটাডেটা ফিল্টার তৈরি (যা ইউজার সাইডবারে সিলেক্ট করেছে)
    # --- ফিক্স: একাধিক field-কে $and + $eq দিয়ে সঠিকভাবে wrap করা হচ্ছে ---
    search_filter = build_chroma_filter(language=doc_lang, type=doc_type)
    
    with st.spinner("Searching and generating response locally..."):
        # মেটাডেটা ফিল্টারসহ ভেক্টর সার্চ চালানো
        docs = db.similarity_search(user_query, k=2, filter=search_filter)
        
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            
            # এলএলএম-এর জন্য প্রম্পট রেডি করা
            prompt_template = ChatPromptTemplate.from_template(
                "Answer the question based only on the following context. If the context doesn't contain the answer, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"
            )
            prompt = prompt_template.format(context=context, question=user_query)
            
            # লোকাল এলএলএম থেকে রেসপন্স জেনারেট
            response = llm.invoke(prompt)
            
            st.markdown("### 🤖 Local RAG Response:")
            st.write(response)
            
            with st.expander("👁️ View Retrieved Context Chunks from DB"):
                for idx, d in enumerate(docs):
                    st.write(f"**Chunk {idx+1} (Source: {d.metadata.get('source')}):**")
                    st.write(d.page_content)
        else:
            st.warning("No documents matched the selected metadata filters. Try adjusting the filters on the left sidebar.")