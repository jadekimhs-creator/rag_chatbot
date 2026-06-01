import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import json, os

from rag import index_document, search_similar, delete_document, generate_answer_stream, generate_graph, extract_pages_from_docx

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="../frontend"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    return FileResponse("../frontend/index.html")

# 문서 메타데이터 파일
DOCS_FILE = "documents.json"

def load_docs():
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_docs(docs):
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


# ── 문서 업로드 ─────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".docx")):
        raise HTTPException(status_code=400, detail="PDF 또는 DOCX 파일만 업로드 가능합니다.")
    
    try:
        file_bytes = await file.read()
        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower()
        
        # 1. 파일 원본 로컬 저장 (뷰어 용도)
        if ext == ".pdf":
            file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
            with open(file_path, "wb") as f:
                f.write(file_bytes)
        elif ext == ".docx":
            # 워드는 원본도 저장하고, 뷰어용 텍스트 프리뷰도 생성
            file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.docx")
            with open(file_path, "wb") as f:
                f.write(file_bytes)
                
            txt_path = os.path.join(UPLOAD_DIR, f"{doc_id}.txt")
            pages = extract_pages_from_docx(file_bytes)
            with open(txt_path, "w", encoding="utf-8") as f:
                for p in pages:
                    f.write(f"--- [페이지 {p['page_number']}] ---\n{p['text']}\n\n")
        
        # 2. 문서 인덱싱 처리
        chunk_count, insights = index_document(doc_id, file.filename, file_bytes, ext)
        
        doc_info = {
            "id": doc_id,
            "filename": file.filename,
            "size": len(file_bytes),
            "chunk_count": chunk_count,
            "summary": insights.get("summary", ""),
            "tags": insights.get("tags", [])
        }
        
        docs = []
        if os.path.exists("documents.json"):
            with open("documents.json", "r", encoding="utf-8") as f:
                docs = json.load(f)
        docs.append(doc_info)
        with open("documents.json", "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False)
            
        return {"message": "업로드 성공", "doc": doc_info}
    except Exception as e:
        print(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 문서 목록 조회 ────────────────────────────────────────────────────────────
@app.get("/documents")
def list_documents():
    return load_docs()


# ── 문서 삭제 ─────────────────────────────────────────────────────────────────
@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    delete_document(doc_id)
    docs = [d for d in load_docs() if d["id"] != doc_id]
    save_docs(docs)
    return {"status": "deleted"}


# ── 채팅 ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    doc_ids: List[str] = []
    is_local: bool = False

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")
    
    try:
        chunks = search_similar(req.question, doc_ids=req.doc_ids)
        
        def event_stream():
            for text_chunk in generate_answer_stream(req.question, chunks, is_local=req.is_local):
                yield text_chunk

        return StreamingResponse(event_stream(), media_type="text/plain")
    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/{doc_id}")
async def get_graph(doc_id: str):
    try:
        graph_data = generate_graph(doc_id)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 프론트엔드 서빙 ────────────────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
