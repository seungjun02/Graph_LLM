import os
import logging
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# --- 경로 및 기본 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
KOSPI100_LIST_PATH = os.path.join(DATA_DIR, 'kospi100_stock_codes.txt')

# --- Azure 서비스 설정 ---
AZURE_DI_ENDPOINT = os.getenv("AZURE_DI_ENDPOINT")
AZURE_DI_KEY = os.getenv("AZURE_DI_KEY")
AZURE_DI_MODEL_ID = "prebuilt-layout"

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
#모델 배포(Model deployments)"를 선택합니다. 여기서 text-embedding-ada-002 모델을 기반으로 생성한 **"배포 이름(Deployment name)"**을 확인하고, 그 이름을 .env 파일에 AZURE_OPENAI_EMBEDDING_DEPLOYMENT="여기에_실제_배포이름_입력" 과 같이 추가
#아무튼 바꿀거면 명시적으로 지정해야됨
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002") 
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "kospi100-rag-index"

# --- RAG 파라미터 ---
TARGET_YEAR = 2024
TARGET_REPORT_CODES = ['11011']
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5

# --- 관계 강도 매핑 ---
STRENGTH_MAPPING = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "None": 0
}

# --- 로깅 설정 ---
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 재시도 설정 ---
RETRY_WAIT_SECONDS = 5
RETRY_ATTEMPTS = 3
