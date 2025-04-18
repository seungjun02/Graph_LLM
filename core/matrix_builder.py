### 10. core/matrix_builder.py  

import numpy as np
import scipy.sparse as sp
from config import logger

def build_adjacency_matrix(relationships: dict) -> sp.coo_matrix:
    """
    LLM에서 추출된 관계 정보를 기반으로 인접 행렬을 생성하는 함수
    relationships: {'company_a': ..., 'company_b': ..., 'relationships': [ { "type": ..., "strength": ... , "evidence": [...] }, ... ]}
    이 예시는 간단히 두 기업 간 단일 관계 강도를 수치화해 인접 행렬에 반영합니다.
    """
    # 예시: 100개 기업 (실제 매핑 정보 필요), 여기서는 임의 크기 100x100 행렬로 생성
    num_nodes = 100
    matrix = np.zeros((num_nodes, num_nodes))
    # 실제 구현 시, 기업 코드와 인덱스 매핑(dict)을 사용하여 각 관계에 해당하는 행렬의 원소를 채웁니다.
    logger.info("인접 행렬 생성 (예시) - 실제 매핑 및 계산 필요")
    # 예시: 두 기업 간 관계 강도를 0.8로 지정 (실제 값은 LLM 결과와 quantification 함수로 계산)
    # matrix[idx_a, idx_b] = 0.8
    return sp.coo_matrix(matrix)

def save_matrix(matrix: sp.coo_matrix, file_path: str):
    try:
        sp.save_npz(file_path, matrix)
        logger.info(f"인접 행렬 저장 완료: {file_path}")
    except Exception as e:
        logger.error(f"인접 행렬 저장 오류: {e}")
