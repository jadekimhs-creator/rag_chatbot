FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 라이브러리 설치 (pdfplumber, docx 등에 필요할 수 있음)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 복사 및 설치
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# 프론트엔드와 백엔드 소스코드 복사
COPY frontend/ ./frontend/
COPY backend/ ./backend/

# 포트 개방
EXPOSE 8000

# 작업 디렉토리를 백엔드로 이동
WORKDIR /app/backend

# FastAPI 서버 실행 (컨테이너 환경)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
