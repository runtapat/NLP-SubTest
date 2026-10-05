# 🛠️ PC-Fix: RAG Chatbot แก้ปัญหา Hardware คอมพิวเตอร์

แชตบอตถาม-ตอบปัญหา Hardware คอมพิวเตอร์ (เปิดไม่ติด จอดำ ร้อนจัด RAM/SSD/PSU/GPU เสีย ฯลฯ) จากคลังเอกสาร **ไทย/อังกฤษ** ด้วยเทคนิค **RAG (Retrieval-Augmented Generation)**

- **Streamlit URL:** `<ใส่ลิงก์หลัง Deploy>`
- **GitHub:** `<ใส่ลิงก์ Repo>`

## แนวคิดของ Domain
ผู้ใช้ทั่วไปมักไม่ทราบสาเหตุเมื่อคอมพิวเตอร์มีปัญหา ระบบนี้ค้นหาวิธีตรวจสอบ/แก้ไขจากเอกสารที่เตรียมไว้ แล้วให้ LLM สรุปเป็นขั้นตอน พร้อมอ้างอิงไฟล์ต้นทาง และตอบว่า **"ไม่พบข้อมูลในเอกสาร"** เมื่อไม่มีคำตอบ เพื่อลดการเดา (hallucination)

## เทคนิค RAG ที่ใช้
| ขั้นตอน | การทำงานใน `app.py` |
|---|---|
| Document Loading & Chunking | โหลด `data/*.txt` → clean text → แบ่งเป็น chunk ~500 ตัวอักษร overlap 80 โดยตัดตามประโยคด้วย PyThaiNLP |
| Embedding & Vector Search | `paraphrase-multilingual-MiniLM-L12-v2` (รองรับไทย/อังกฤษ) + FAISS `IndexFlatIP` (cosine), Top-K = 4 |
| Prompt Engineering | System prompt บังคับตอบจาก CONTEXT เท่านั้น, อ้างอิง `[ชื่อไฟล์]`, ไม่มีข้อมูลให้ตอบ "ไม่พบข้อมูลในเอกสาร" |
| LLM | Groq API (`llama-3.3-70b-versatile`), temperature 0.1 |
| Chatbot Interface | Streamlit chat, จำบทสนทนาล่าสุด 3 รอบ, แสดงเอกสารอ้างอิงพร้อมคะแนน similarity ทุกคำตอบ |

## วิธีรันบนเครื่อง
```bash
pip install -r requirements.txt
mkdir .streamlit && echo 'GROQ_API_KEY = "gsk_xxx"' > .streamlit/secrets.toml   # ไฟล์นี้ถูก .gitignore ไว้
streamlit run app.py
```
ถ้าไม่ตั้งค่า key แอปจะแสดงเฉพาะผลการค้นหาเอกสาร (ไม่เรียก LLM)

## Deploy บน Streamlit Community Cloud
1. Push โค้ดขึ้น GitHub (**ห้าม** commit API key)
2. share.streamlit.io → New app → เลือก repo/branch และไฟล์ `app.py`
3. Advanced settings → Secrets → ใส่ `GROQ_API_KEY = "gsk_xxx"`

## แหล่งที่มาของเอกสาร
ไฟล์ใน `data/` (12 ไฟล์ ~16,500 ตัวอักษร) เป็น **ข้อมูลจำลองที่สร้างด้วย AI** สรุปจากความรู้ทั่วไปด้านการซ่อมบำรุงคอมพิวเตอร์ เพื่อการศึกษาเท่านั้น ไม่ใช่เอกสารทางการของผู้ผลิต

## การประเมินผล
`test_questions.csv` มีคำถามทดสอบ 12 ข้อ (9 ข้อมีคำตอบในเอกสาร, 3 ข้อไม่มี เพื่อทดสอบการตอบ "ไม่พบข้อมูล")

## ตัวอย่าง Prompt ที่ใช้สั่ง AI ช่วยเขียนโค้ด
1. "ช่วยเขียน Streamlit app ระบบ RAG: โหลดไฟล์ .txt จากโฟลเดอร์ data/, chunk ~500 ตัวอักษร, embed ด้วย sentence-transformers แบบ multilingual, ค้นด้วย FAISS, เรียก Groq API โดยอ่านคีย์จาก st.secrets"
2. "เขียน system prompt ให้ LLM ตอบจาก context เท่านั้น อ้างอิงชื่อไฟล์ และตอบว่า 'ไม่พบข้อมูลในเอกสาร' เมื่อไม่มีคำตอบ"
3. "สร้างเอกสารจำลองภาษาไทย/อังกฤษเกี่ยวกับการแก้ปัญหา Hardware คอมพิวเตอร์ 12 ไฟล์ และชุดคำถามทดสอบ 12 ข้อ โดยมี 3 ข้อที่ไม่มีคำตอบในเอกสาร"

## โครงสร้างไฟล์
```
app.py · requirements.txt · README.md · test_questions.csv · .gitignore
data/01_no_power.txt ... data/12_maintenance_warranty.txt
```
