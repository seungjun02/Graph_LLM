#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
메인 파이프라인 실행 오케스트레이션
"""

from core import dart_utils, document_processor, vector_store_manager, llm_extractor, matrix_builder
from config import logger, KOSPI100_LIST_PATH, TARGET_YEAR , DATA_DIR
import os


# PDF 파일이 저장된 디렉토리 경로 정의 (config.py에 추가하거나 여기서 직접 정의)
INPUT_PDF_DIR = os.path.join(DATA_DIR, 'input_pdfs')

def run_pipeline():
    logger.info("==========================================")
    logger.info("KOSPI 100 기업 관계 RAG 파이프라인 실행 시작")
    logger.info("==========================================")
    
    # 1. 입력 데이터 로드 (종목 코드 목록)
    if not os.path.exists(KOSPI100_LIST_PATH):
        logger.error(f"KOSPI 100 목록 파일을 찾을 수 없습니다: {KOSPI100_LIST_PATH}")
        return
    with open(KOSPI100_LIST_PATH, 'r', encoding='utf-8') as f:
        stock_codes = [line.strip() for line in f if line.strip()]
    logger.info(f"종목 코드 {len(stock_codes)}개 로드됨.")



    '''
    # 2. DART 보고서 다운로드 및 처리 (dart_downloads 폴더에 저장)
    # 예시: 첫 번째 종목에 대해 처리 (실제 구현 시 루프 처리)
    sample_corp_code = stock_codes[0]
    report_no = dart_utils.find_latest_annual_report(sample_corp_code, TARGET_YEAR)
    if report_no:
        download_path = os.path.join("dart_downloads", sample_corp_code)
        zip_file = dart_utils.download_report_file(report_no, download_path)
        if zip_file:
            html_bytes = dart_utils.extract_html_from_zip(zip_file)
            if html_bytes:
                html_content = dart_utils.decode_html(html_bytes)
            else:
                html_content = ""
        else:
            html_content = ""
    else:
        html_content = ""
'''


    # 2-1 전체 기업 처리를 위한 루프 (예시: stock_codes 리스트 사용)
    for stock_code in stock_codes:
        logger.info(f"--- 처리 시작: {stock_code} ---")

        # 2. 로컬 PDF 파일 로드
        pdf_file_path = os.path.join(INPUT_PDF_DIR, f"{stock_code}.pdf")
        pdf_content_bytes = None # 초기화

        if os.path.exists(pdf_file_path):
            try:
                with open(pdf_file_path, "rb") as f: # PDF는 바이너리 모드('rb')로 읽어야 함
                    pdf_content_bytes = f.read()
                logger.info(f"입력 PDF 파일 로드 성공: {pdf_file_path}")
            except Exception as e:
                logger.error(f"PDF 파일 읽기 오류 ({stock_code}): {e}")
                # 오류 발생 시 해당 기업 처리를 건너뛸 수 있음
                continue
        else:
            logger.warning(f"입력 PDF 파일을 찾을 수 없습니다: {pdf_file_path}. 해당 기업 처리 건너뜀.")
            # 파일이 없으면 해당 기업 처리를 건너뜀
            continue


        # 3. 문서 처리: Azure Document Intelligence로 분석 (PDF 바이트 입력)
        # pdf_content_bytes가 정상적으로 로드되었을 때만 실행
        if pdf_content_bytes:
            # analyze_document_content 함수는 바이트 입력을 처리할 수 있어야 함
            # (현재 document_processor.py의 구현을 보면 가능해 보임)
            di_result = document_processor.analyze_document_content(pdf_content_bytes)

            # DI 결과가 있는지 확인 후 섹션 추출
            if di_result:
                sections = document_processor.extract_sections_from_di_result(di_result)
                logger.info(f"추출된 섹션: {list(sections.keys())}")

                # 4. 텍스트 임베딩 및 벡터 저장
                full_text = "\n\n".join(sections.values())
                if full_text.strip(): # 추출된 텍스트가 있을 경우에만 진행
                    text_chunks = vector_store_manager.get_text_chunks(full_text)
                    embeddings = [vector_store_manager.get_embedding(chunk) for chunk in text_chunks if chunk] # 빈 청크 제외
                    # 실제 Azure Search 업로드 구현 필요
                    vector_store_manager.upsert_documents_to_ai_search(text_chunks, embeddings)
                else:
                    logger.warning(f"{stock_code}: DI 결과에서 유효한 텍스트를 추출하지 못했습니다.")

            else:
                logger.warning(f"{stock_code}: Document Intelligence 분석 결과가 없습니다.")
        else:
            # pdf_content_bytes가 None이면 이미 로그가 찍혔으므로 여기서는 별도 처리 불필요
            pass

        logger.info(f"--- 처리 완료: {stock_code} ---")
        # --- 루프는 여기까지 ---






    '''
    # 3. 문서 처리: Azure Document Intelligence로 분석 (Markdown 형태로 출력)
    di_result = document_processor.analyze_document_content(html_content)
    sections = document_processor.extract_sections_from_di_result(di_result, target_sections=["사업의 내용", "주주에 관한 사항"])
    logger.info(f"추출된 섹션: {list(sections.keys())}")
'''



    '''
    # 4. 텍스트 임베딩 및 벡터 저장: 임베딩 생성 후, Azure AI Search 인덱스에 업로드
    full_text = "\n\n".join(sections.values())
    text_chunks = vector_store_manager.get_text_chunks(full_text)
    embeddings = [vector_store_manager.get_embedding(chunk) for chunk in text_chunks]
    vector_store_manager.upsert_documents_to_ai_search(text_chunks, embeddings)
'''




    # 5. LLM 기반 관계 추출: 예시로 두 기업 간 관계 분석
    prompt = llm_extractor.create_relationship_extraction_prompt(
        context_chunks=text_chunks,
        company_a_name="삼성전자",
        company_b_name="삼성SDI"
    )
    llm_response = llm_extractor.extract_relationship_with_llm(prompt)
    relationships = llm_extractor.parse_llm_json_output(llm_response)
    
    # 6. 인접 행렬 생성: 추출된 관계 정보를 기반으로 그래프 생성
    matrix = matrix_builder.build_adjacency_matrix(relationships)
    matrix_builder.save_matrix(matrix, "output_matrices/adjacency_matrix.npz")
    
    logger.info("==========================================")
    logger.info("파이프라인 실행 완료")
    logger.info("==========================================")


# --- 메인 실행 블록 수정 ---
if __name__ == '__main__':
    # 1. 로그 시작
    logger.info("===== 프로그램 실행 시작 =====")

    # 2. Azure AI Search 인덱스 설정 확인 및 생성/업데이트 시도
    logger.info("--- Azure AI Search 인덱스 설정 확인/시도 ---")
    # create_search_index 함수는 성공 시 True, 실패 시 False를 반환하도록 수정했음 (이전 답변 참고)
    index_ready = create_search_index()

    # 3. 인덱스가 준비되었는지 확인 후 메인 파이프라인 실행
    if index_ready:
        logger.info("--- 인덱스 준비 완료. 메인 파이프라인 시작 ---")
        try:
            run_pipeline() # 메인 파이프라인 실행
            logger.info("--- 메인 파이프라인 성공적으로 완료 ---")
        except Exception as e:
            logger.error(f"--- 메인 파이프라인 실행 중 오류 발생: {e} ---")
            import traceback
            traceback.print_exc() # 상세 오류 스택 출력
    else:
        logger.error("--- 인덱스 설정 실패. 메인 파이프라인을 실행할 수 없습니다. ---")

    logger.info("===== 프로그램 실행 종료 =====")