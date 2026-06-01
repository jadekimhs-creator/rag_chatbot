import os
import io
import json
import numpy as np
import pdfplumber
import docx
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

STORE_PATH = "vector_store.json"


# ── 순수 Python 벡터 스토어 ─────────────────────────────────────────────────
class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self._load()

    def _load(self):
        if os.path.exists(STORE_PATH):
            with open(STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", [])
                self.embeddings = data.get("embeddings", [])
                self.metadatas = data.get("metadatas", [])

    def _save(self):
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "metadatas": self.metadatas
            }, f, ensure_ascii=False)

    def add(self, documents, embeddings, metadatas):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self._save()

    def query(self, query_embedding, n_results=5, doc_ids=None, session_id="default"):
        if not self.embeddings:
            return []
        q = np.array(query_embedding)
        embs = np.array(self.embeddings)
        sims = np.dot(embs, q) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q) + 1e-10)
        
        for i, m in enumerate(self.metadatas):
            if m.get("session_id", "default") != session_id:
                sims[i] = -1
            elif doc_ids and m.get("doc_id") not in doc_ids:
                sims[i] = -1
                
        top = np.argsort(sims)[::-1][:n_results]
        
        # 반환할 때 메타데이터(페이지 번호 등)도 함께 반환
        return [{"text": self.documents[i], "metadata": self.metadatas[i]} for i in top if sims[i] > 0]

    def delete(self, doc_id):
        keep = [i for i, m in enumerate(self.metadatas) if m.get("doc_id") != doc_id]
        self.documents = [self.documents[i] for i in keep]
        self.embeddings = [self.embeddings[i] for i in keep]
        self.metadatas = [self.metadatas[i] for i in keep]
        self._save()


store = VectorStore()


def extract_pages_from_pdf(file_bytes: bytes) -> list[dict]:
    pages = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({"page_number": i + 1, "text": text})
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return pages


def extract_pages_from_docx(file_bytes: bytes) -> list[dict]:
    pages = []
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        # 워드는 물리적 페이지 구분이 어려우므로 1000자 단위로 가상 페이지 분할
        chunk_size = 1000
        page_num = 1
        for i in range(0, len(full_text), chunk_size):
            pages.append({
                "page_number": page_num,
                "text": full_text[i:i+chunk_size]
            })
            page_num += 1
        if not pages:
            pages.append({"page_number": 1, "text": "내용 없음"})
    except Exception as e:
        print(f"DOCX extraction error: {e}")
    return pages


# ── 텍스트 청킹 ──────────────────────────────────────────────────────────────
def split_text(text: str, chunk_size=800, overlap=100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ── 임베딩 생성 (새 SDK) ─────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return response.embeddings[0].values


def get_query_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return response.embeddings[0].values


# ── Auto-Insights (요약 & 태그 추출) ─────────────────────────────────────────
def generate_insights(full_text: str) -> dict:
    prompt = f"""다음 문서를 분석하여 3줄 이내의 핵심 요약과 5개의 주요 키워드 해시태그를 JSON 형식으로 반환하세요.
형식: {{"summary": "...", "tags": ["#태그1", "#태그2", ...]}}

문서 내용:
{full_text[:15000]} # 너무 길면 앞부분만 분석
"""
    try:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        elif text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Insight generation error: {e}")
        return {"summary": "요약을 생성할 수 없습니다.", "tags": []}


# ── 문서 인덱싱 ──────────────────────────────────────────────────────────────
def index_document(doc_id: str, filename: str, file_bytes: bytes, file_ext: str = ".pdf", session_id: str = "default") -> tuple[int, dict]:
    if file_ext == ".docx":
        pages = extract_pages_from_docx(file_bytes)
    else:
        pages = extract_pages_from_pdf(file_bytes)
        
    full_text = ""
    
    docs, embeddings, metadatas = [], [], []
    for page in pages:
        full_text += page["text"] + "\n"
        chunks = split_text(page["text"])
        for i, chunk in enumerate(chunks):
            embeddings.append(get_embedding(chunk))
            docs.append(chunk)
            metadatas.append({
                "doc_id": doc_id, 
                "filename": filename, 
                "page_number": page["page_number"],
                "chunk_index": i,
                "session_id": session_id
            })
            
    store.add(docs, embeddings, metadatas)
    
    # Insights 생성
    insights = generate_insights(full_text)
    return len(docs), insights


# ── 유사 청크 검색 (다중 문서 지원) ────────────────────────────────────────────
def search_similar(query: str, doc_ids: list[str] = None, n_results: int = 5, session_id: str = "default") -> list[dict]:
    q_emb = get_query_embedding(query)
    return store.query(q_emb, n_results=n_results, doc_ids=doc_ids, session_id=session_id)


# ── 마인드맵 (지식 그래프) 생성 ─────────────────────────────────────────────
def generate_graph(doc_id: str) -> dict:
    # 해당 문서의 모든 텍스트 모으기 (최대 길이 제한)
    full_text = ""
    for i, m in enumerate(store.metadatas):
        if m.get("doc_id") == doc_id:
            full_text += store.documents[i] + "\n"
    
    prompt = f"""다음 문서를 분석하여 지식 마인드맵을 그리기 위한 노드(개념)와 엣지(관계)를 JSON 형식으로 추출하세요.
형식: {{"nodes": [{{"id": 1, "label": "개념이름", "group": "분류"}}], "edges": [{{"from": 1, "to": 2, "label": "관계설명"}}]}}
중요한 핵심 개념 10개 내외만 추출하세요.

문서 내용:
{full_text[:15000]}
"""
    try:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        elif text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Graph generation error: {e}")
        return {"nodes": [], "edges": []}


# ── 문서 삭제 ────────────────────────────────────────────────────────────────
def delete_document(doc_id: str):
    store.delete(doc_id)


# ── Gemini / Ollama 답변 생성 (출처 표시 & 웹 검색 확장) ──────────────────────────────
def generate_answer_stream(question: str, chunks_data: list[dict], is_local: bool = False):
    context_str = ""
    for data in chunks_data:
        page = data['metadata'].get('page_number', 1)
        context_str += f"[문서 페이지: {page}]\n{data['text']}\n\n---\n\n"
        
    prompt = f"""당신은 주어진 문서를 기반으로 질문에 답하는 전문 AI 어시스턴트입니다.

아래는 업로드된 문서에서 찾은 관련 내용입니다:
{context_str}

---

위 내용을 바탕으로 다음 질문에 정확하고 친절하게 답하세요.
답변을 할 때 참조한 내용이 있다면 반드시 "[{page}]" 형식으로 페이지 번호 출처를 달아주세요. (예: 이 내용은 이러합니다. [3])
만약 위 문서에 질문에 대한 내용이 전혀 없다면, 당신이 가진 지식과 **구글 웹 검색**을 활용하여 답변하세요. 
문서를 참조했는지, 웹을 참조했는지 명확하게 알 수 있도록 대답해주세요.
답변은 한국어로 작성하세요.

질문: {question}

답변:"""

    if is_local:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": "llama3", "prompt": prompt}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req) as response:
                for line in response:
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if "response" in data:
                            yield data["response"]
        except urllib.error.URLError:
            yield "❌ 로컬 모드(Ollama)에 연결할 수 없습니다. Ollama 앱이 실행 중이고 'llama3' 모델이 설치되어 있는지 확인해주세요. (설치: ollama run llama3)"
        return

    response = client.models.generate_content_stream(
        model="gemini-flash-lite-latest",
        contents=prompt
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text
