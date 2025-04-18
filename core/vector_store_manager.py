from config import CHUNK_SIZE, CHUNK_OVERLAP, logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from openai import OpenAI, AzureOpenAI # AzureOpenAI 추가 (선택 사항이지만 권장)
from config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT, logger # AZURE_OPENAI_EMBEDDING_DEPLOYMENT 임포트 확인!
)
import traceback # 상세 오류 로깅을 위해 추가


import os
import uuid # 고유 ID 생성을 위해 추가
from config import AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX_NAME # Search 설정 추가
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Azure Search 관련 임포트 추가
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


def get_text_chunks(text: str) -> list:
    if not text:
        return []
    try:
        separators = ["\n## ", "\n### ", "\n\n", "\n", " "]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=separators
        )
        chunks = splitter.split_text(text)
        logger.info(f"텍스트 청크 분할 완료: 총 {len(chunks)}개")
        return chunks
    except Exception as e:
        logger.error(f"텍스트 분할 오류: {e}")
        return []

# (선택 사항) 클라이언트 초기화 개선: AzureOpenAI 사용
def initialize_openai_client():
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION]):
        logger.error("Azure OpenAI 설정(Endpoint, Key, API Version)이 누락되었습니다.")
        return None
    try:
        # Azure 전용 클라이언트 사용 권장
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )
        # 또는 기존 방식:
        # client = OpenAI(api_key=AZURE_OPENAI_KEY, api_base=AZURE_OPENAI_ENDPOINT, api_version=AZURE_OPENAI_API_VERSION)
        logger.info("Azure OpenAI 클라이언트 초기화 성공")
        return client
    except Exception as e:
        logger.error(f"OpenAI 클라이언트 초기화 오류: {e}")
        return None

# get_embedding 함수 확인 및 수정
def get_embedding(text_chunk: str) -> list:
    # (개선 제안) 클라이언트를 매번 초기화하는 대신, 한 번 초기화해서 재사용하는 것이 효율적입니다.
    # 예: 노트북 셀 시작 부분에서 client = initialize_openai_client() 호출 후,
    # get_embedding 함수는 client를 인자로 받도록 수정 -> def get_embedding(text_chunk: str, client) -> list:
    client = initialize_openai_client() # 현재 코드는 매번 초기화

    if not client:
        logger.error("OpenAI 클라이언트가 유효하지 않아 임베딩을 생성할 수 없습니다.")
        return []
    if not text_chunk or not text_chunk.strip():
        logger.warning("내용이 없는 청크는 임베딩을 건너뜁니다.")
        return []

    try:
        # --- 중요: model 파라미터에 배포 이름(Deployment Name) 사용 ---
        response = client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT, # config에서 가져온 배포 이름 사용!
            input=text_chunk
        )
        # --- 중요: 결과 접근 방식 수정 ---
        # embedding = response.data.embedding # 이전 방식 (오류 발생 가능)
        embedding = response.data[0].embedding # 수정된 방식 (리스트의 첫 항목에서 벡터 추출)
        return embedding

    except Exception as e:
        # --- 오류 로깅 개선 ---
        logger.error(f"임베딩 생성 중 오류 발생 (청크 앞부분: '{text_chunk[:100]}...'): {type(e).__name__} - {e}")
        # 상세 스택 트레이스가 필요하면 아래 주석 해제
        # traceback.print_exc()
        return [] # 오류 발생 시 빈 리스트 반환


# --- Azure AI Search 클라이언트 초기화 함수 추가 ---
def initialize_search_client():
    """Initializes and returns an Azure AI Search client."""
    # 설정 값 존재 여부 확인
    if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX_NAME]):
        logger.error("Azure AI Search 설정(Endpoint, Key, Index Name)이 config.py 또는 .env 파일에 누락되었습니다.")
        return None
    try:
        # API 키를 사용하여 자격 증명 생성
        credential = AzureKeyCredential(AZURE_SEARCH_KEY)
        # SearchClient 인스턴스 생성
        search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT,
                                     index_name=AZURE_SEARCH_INDEX_NAME,
                                     credential=credential)
        logger.info(f"Azure AI Search 클라이언트 초기화 성공 (인덱스: {AZURE_SEARCH_INDEX_NAME})")
        return search_client
    except Exception as e:
        logger.error(f"Azure AI Search 클라이언트 초기화 실패: {e}")
        return None

# --- 기존 placeholder 함수를 실제 구현으로 대체 ---
# def upsert_documents_to_ai_search(text_chunks, embeddings):
#     logger.info("Azure AI Search에 문서 업로드 (예시) - 실제 구현 필요")

def upsert_documents_to_ai_search(text_chunks: list[str], embeddings: list[list[float]], source_document_id: str = None):
    """
    텍스트 청크와 임베딩을 Azure AI Search 인덱스에 업로드(업서트)합니다.

    Args:
        text_chunks: 업로드할 텍스트 청크 리스트.
        embeddings: 각 텍스트 청크에 해당하는 임베딩 벡터 리스트.
        source_document_id: (선택 사항) 이 청크들의 출처 문서 식별자 (예: 파일명, rcept_no).
                           인덱스 스키마에 해당 필드가 정의되어 있어야 합니다.
    """
    search_client = initialize_search_client()
    if not search_client:
        logger.error("Search client 초기화 실패로 문서 업로드를 중단합니다.")
        return False # 실패를 나타내는 값 반환

    # 입력 데이터 유효성 검사
    if len(text_chunks) != len(embeddings):
        logger.error(f"텍스트 청크({len(text_chunks)}개)와 임베딩({len(embeddings)}개)의 개수가 일치하지 않습니다.")
        return False
    if not text_chunks:
        logger.warning("업로드할 문서(청크)가 없습니다.")
        return True # 처리할 내용이 없으므로 성공으로 간주

    documents_to_upload = []
    logger.info(f"Azure AI Search 업로드 준비: 총 {len(text_chunks)}개 문서")

    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        # --- 중요: 아래 딕셔너리 키는 실제 Azure AI Search 인덱스 필드명과 일치해야 합니다 ---
        document = {
            "id": str(uuid.uuid4()),  # 각 청크 문서에 고유한 ID 부여 (필수)
            "content": chunk,         # 텍스트 내용 필드 (예: 필드명 'content')
            "content_vector": embedding, # 임베딩 벡터 필드 (예: 필드명 'content_vector')
             # --- 선택적 메타데이터 필드 (인덱스에 해당 필드가 정의되어 있어야 함) ---
            "source_document": source_document_id or "unknown", # 출처 문서 식별자 필드 (예: 필드명 'source_document')
            "chunk_index": i          # 문서 내 청크 순서 필드 (예: 필드명 'chunk_index')
        }
        documents_to_upload.append(document)

    try:
        # 문서를 인덱스에 업로드 (기본적으로 upsert)
        # 많은 문서를 업로드할 경우, 배치(batch)로 나누어 처리하는 것이 더 안정적일 수 있습니다.
        # 예: batch_size = 1000; for i in range(0, len(documents), batch_size): batch = documents[i:i+batch_size]; result = client.upload...
        result = search_client.upload_documents(documents=documents_to_upload)

        # 결과 확인
        successful_uploads = sum(1 for r in result if r.succeeded)
        logger.info(f"Azure AI Search 문서 업로드 완료: {successful_uploads} / {len(documents_to_upload)} 성공")

        # 실패한 경우 로그 남기기 (선택 사항)
        if successful_uploads < len(documents_to_upload):
            failure_count = 0
            for idx, res in enumerate(result):
                if not res.succeeded:
                    logger.warning(f"문서 업로드 실패: Index={idx}, Key={res.key}, ErrorCode={res.status_code}, ErrorMsg={res.error_message}")
                    failure_count += 1
                    if failure_count >= 5: # 너무 많은 실패 로그 방지
                         logger.warning("... 추가 실패 로그 생략 ...")
                         break
            return False # 일부 실패 시 False 반환

        return True # 모든 문서 처리 완료 시 True 반환

    except Exception as e:
        logger.error(f"Azure AI Search 문서 업로드 중 오류 발생: {e}")
        return False # 오류 발생 시 False 반환

