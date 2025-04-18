# kospi_relation_rag_pipeline/core/index_setup.py

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField, # ComplexField는 사용하지 않으므로 제거해도 됨
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField, # SimpleField는 SearchField로 대체 가능
    SearchableField, # SearchableField는 SearchField로 대체 가능
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField
)
# config.py에서 설정값 가져오기
try:
    from config import AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX_NAME, logger, AZURE_OPENAI_EMBEDDING_DEPLOYMENT
except ModuleNotFoundError:
    print("ERROR: config.py 파일을 찾을 수 없습니다. 프로젝트 루트에서 실행 중인지 확인하세요.")
    exit()
except ImportError as e:
    print(f"ERROR: config.py에서 설정을 가져오는 중 오류 발생: {e}")
    exit()


# --- 인덱스 설정 상수 ---
# 임베딩 모델 차원 수 (text-embedding-ada-002 기준)
# 중요: 사용하는 임베딩 모델에 맞춰 수정 필요!
# 예: 다른 모델이 3072 차원이라면 EMBEDDING_DIMENSIONS = 3072 로 변경
EMBEDDING_DIMENSIONS = 1536
# 벡터 검색 프로필 및 알고리즘 이름 정의
VECTOR_PROFILE_NAME = "my-hnsw-profile"
HNSW_ALGORITHM_NAME = "my-hnsw-config"
# 의미 체계 검색 구성 이름 정의
SEMANTIC_CONFIG_NAME = "my-semantic-config"

def create_search_index():
    """Azure AI Search 인덱스를 생성하거나 업데이트합니다."""

    # 설정 값 확인
    if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX_NAME]):
        logger.error("Azure AI Search 설정(Endpoint, Key, Index Name)이 config.py 또는 .env 파일에 누락되어 인덱스를 생성할 수 없습니다.")
        return False # 실패 반환

    logger.info(f"Azure AI Search 인덱스 '{AZURE_SEARCH_INDEX_NAME}' 생성 또는 업데이트 시도...")
    try:
        credential = AzureKeyCredential(AZURE_SEARCH_KEY)
        # SearchIndexClient는 인덱스 관리용 클라이언트입니다.
        index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=credential)

        # 인덱스 필드 정의
        fields = [
            # Edm.String 타입, key=True 필수
            SearchField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=False),
            # Edm.String 타입, 검색 가능, 검색 결과 반환 가능
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True, filterable=False, sortable=False, facetable=False, retrievable=True),
            # 벡터 필드: Collection(Edm.Single) 타입, 검색 가능, 차원 수, 프로필 이름 지정
            SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, retrievable=True, # 벡터 검색을 위해 searchable=True
                        vector_search_dimensions=EMBEDDING_DIMENSIONS,
                        vector_search_profile_name=VECTOR_PROFILE_NAME),
            # 메타데이터 필드들 (필요에 따라 속성 조정)
            SearchField(name="source_document", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True, retrievable=True),
            SearchField(name="chunk_index", type=SearchFieldDataType.Int32, sortable=True, filterable=True, facetable=False, retrievable=True)
        ]

        # 벡터 검색 구성 정의 (HNSW 알고리즘 사용 예시)
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name=HNSW_ALGORITHM_NAME)], # 사용할 알고리즘 설정 정의
            profiles=[VectorSearchProfile(name=VECTOR_PROFILE_NAME, algorithm_configuration_name=HNSW_ALGORITHM_NAME)] # 프로필 정의 및 알고리즘 연결
        )

        # 의미 체계 검색(Semantic Search) 구성 정의 (랭킹 성능 향상에 도움)
        semantic_search = SemanticSearch(
             configurations=[
                 SemanticConfiguration(
                     name=SEMANTIC_CONFIG_NAME,
                     prioritized_fields=SemanticPrioritizedFields(
                         title_field=None, # 청크 데이터에는 별도 제목 필드가 없으므로 None
                         content_fields=[SemanticField(field_name="content")] # 'content' 필드를 의미 체계 분석 대상으로 지정
                     )
                 )
             ]
        )

        # SearchIndex 객체 생성
        index = SearchIndex(
            name=AZURE_SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search # 의미 체계 검색 설정 추가
        )

        # 인덱스 생성 또는 업데이트 실행
        result = index_client.create_or_update_index(index)
        logger.info(f"Azure AI Search 인덱스 '{result.name}' 생성 또는 업데이트 완료.")
        return True # 성공 반환

    except Exception as e:
        logger.error(f"Azure AI Search 인덱스 생성 또는 업데이트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc() # 상세 오류 출력
        return False # 실패 반환
