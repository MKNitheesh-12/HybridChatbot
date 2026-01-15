import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("🔍 TESTING YOUR SETUP")
print("=" * 50)

# 1. Check API Key
print("\n1️⃣ Checking API Key...")
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    print(f"✅ API Key found: {api_key[:10]}...")
else:
    print("❌ API Key NOT found!")

# 2. Check Files
print("\n2️⃣ Checking Files...")

files_to_check = {
    "NLP_QA_Pairs12.xlsx": "Excel file",
    "pdfs": "PDF directory",
    "faiss_index": "FAISS index"
}

for file, desc in files_to_check.items():
    if os.path.exists(file):
        if os.path.isdir(file):
            contents = os.listdir(file)
            print(f"✅ {desc} exists with {len(contents)} items")
        else:
            print(f"✅ {desc} exists")
    else:
        print(f"❌ {desc} NOT found!")

# 3. Test Excel Loading
print("\n3️⃣ Testing Excel Loading...")
try:
    import pandas as pd
    df = pd.read_excel("NLP_QA_Pairs12.xlsx")
    print(f"✅ Excel loaded: {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ Excel error: {e}")

# 4. Test Embeddings
print("\n4️⃣ Testing Embeddings...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    test_embed = embeddings.embed_query("test")
    print(f"✅ Embeddings working: {len(test_embed)} dimensions")
except Exception as e:
    print(f"❌ Embeddings error: {e}")

# 5. Test Groq API
print("\n5️⃣ Testing Groq API...")
try:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print(f"✅ Groq API working: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Groq API error: {e}")

print("\n" + "=" * 50)
print("✅ TESTING COMPLETE")
print("=" * 50)