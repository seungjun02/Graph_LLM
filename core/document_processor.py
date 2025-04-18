from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from config import AZURE_DI_ENDPOINT, AZURE_DI_KEY, AZURE_DI_MODEL_ID, logger
from tenacity import retry, stop_after_attempt, wait_fixed

def initialize_di_client():
    if not AZURE_DI_ENDPOINT or not AZURE_DI_KEY:
        logger.error("Azure DI 설정이 누락되었습니다.")
        return None
    try:
        credential = AzureKeyCredential(AZURE_DI_KEY)
        client = DocumentIntelligenceClient(endpoint=AZURE_DI_ENDPOINT, credential=credential)
        logger.info("DI 클라이언트 초기화 성공")
        return client
    except Exception as e:
        logger.error(f"DI 클라이언트 초기화 실패: {e}")
        return None

# analyze_pdf_document 함수 수정 (또는 analyze_document_content 함수)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
# def analyze_pdf_document(pdf_file_path: str, model_id=AZURE_DI_MODEL_ID, output_format=ContentFormat.MARKDOWN): # 이전 버전
def analyze_pdf_document(pdf_file_path: str, model_id=AZURE_DI_MODEL_ID, output_format: str = "markdown"): # 수정: 타입 힌트 및 기본값 변경
    """지정된 경로의 PDF 파일을 Azure Document Intelligence로 분석합니다."""
    client = initialize_di_client()
    if not client or not pdf_file_path or not os.path.exists(pdf_file_path):
        logger.error(f"DI 클라이언트가 초기화되지 않았거나 PDF 파일 경로가 유효하지 않습니다: {pdf_file_path}")
        return None
    try:
        logger.info(f"PDF 파일 분석 시작: {pdf_file_path}")
        with open(pdf_file_path, "rb") as pdf_file:
            poller = client.begin_analyze_document(
                model_id,      # 첫 번째 위치 인자
                pdf_file,      # 두 번째 위치 인자 (analyze_request에 해당)
                # output_content_format 등 다른 파라미터는 키워드 인자로 전달 가능
                output_content_format=output_format
                # locale="ko-KR" # 필요시 추가
            )
        result = poller.result()
        logger.info(f"DI 분석 완료: {pdf_file_path}")
        return result
    except Exception as e:
        logger.error(f"DI 분석 오류 ({pdf_file_path}): {e}")
        return None # 오류 시 None 반환

# 만약 analyze_document_content 함수도 사용한다면 동일하게 수정
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def analyze_document_content(doc_content: bytes, model_id=AZURE_DI_MODEL_ID, output_format: str = "markdown"):
    """주어진 바이트(bytes) 데이터를 Azure Document Intelligence로 분석합니다."""
    client = initialize_di_client()
    if not client or not doc_content:
        logger.error("DI 클라이언트가 초기화되지 않았거나 분석할 콘텐츠가 없습니다.")
        return None
    try:
        logger.info(f"바이트 데이터 분석 시작 (콘텐츠 길이: {len(doc_content)})")
        poller = client.begin_analyze_document(
            model_id,      # 첫 번째 위치 인자
            doc_content,   # 두 번째 위치 인자 (analyze_request에 해당)
            output_content_format=output_format
        )
        result = poller.result()
        logger.info("DI 분석 완료 (바이트 데이터)")
        return result
    except Exception as e:
        logger.error(f"DI 분석 오류 (바이트 데이터): {e}")
        return None

# kospi_relation_rag_pipeline/core/document_processor.py

def extract_sections_from_di_result(di_result): # target_sections 인자 제거
    import re
    if not di_result or not di_result.content:
        logger.warning("분석 결과가 없습니다.")
        return {}
    markdown_content = di_result.content
    extracted = {}
    lines = markdown_content.splitlines()
    current_title = None
    current_content = []
    # H2, H3 제목 패턴은 그대로 사용
    header_pattern = re.compile(r"^(#{2,3})\s+(.*)")
    for line in lines:
        match = header_pattern.match(line)
        if match:
            # 이전 섹션이 있었다면 (제목 필터링 없이) 저장
            if current_title:
                # --- 필터링 조건 제거 ---
                # if any(t in current_title for t in target_sections): # 이 조건 제거
                extracted[current_title] = "\n".join(current_content).strip()
            # --------------------------
            current_title = match.group(2).strip() # 새 섹션 제목
            current_content = [] # 새 섹션 내용 초기화
        else:
            # 현재 섹션의 내용 추가
            if current_title:
                current_content.append(line)

    # 마지막 섹션 저장 (제목 필터링 없이)
    if current_title:
        # --- 필터링 조건 제거 ---
        # if any(t in current_title for t in target_sections): # 이 조건 제거
        extracted[current_title] = "\n".join(current_content).strip()
    # --------------------------

    # 추출된 섹션 수 로깅
    if extracted:
        logger.info(f"Markdown H2/H3 기준 섹션 추출 완료: 총 {len(extracted)}개")
    else:
        logger.warning("Markdown H2/H3 헤더를 찾지 못했거나 내용이 없습니다.")

    return extracted