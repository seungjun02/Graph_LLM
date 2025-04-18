import re
import json
from config import AZURE_OPENAI_CHAT_DEPLOYMENT, logger
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI  # Azure OpenAI 클라이언트

def create_relationship_extraction_prompt(context_chunks: list[str], company_a_name: str, company_b_name: str) -> str:
    """
    주어진 컨텍스트와 두 회사 이름을 바탕으로 경쟁, 소유, 공급 관계 연결 여부 및
    근거 추출을 위한 LLM 프롬프트를 생성합니다.
    """
    context_str = "\n\n---\n\n".join(context_chunks)
    if not context_str.strip():
        context_str = "제공된 컨텍스트가 없습니다."

    # 분석 대상 관계 유형 정의
    target_relation_types = ["Competition", "Ownership", "Supply"]

    # LLM에게 요청할 JSON 출력 스키마 정의 (연결 여부 + 근거)
    json_schema = f"""
{{
  "company_a": "{company_a_name}",
  "company_b": "{company_b_name}",
  "relationships": [
    # 아래 형식으로 {target_relation_types} 각 유형에 대한 결과를 반드시 포함하세요.
    {{
      "type": "string",  # {target_relation_types} 중 하나
      "connected": boolean, # 관계가 존재하면 true, 아니면 false
      "evidence": ["string", "..."] # 관계가 존재하면(true) 근거 문장(최대 2개), 없으면 빈 리스트 []
    }}
  ]
}}
    """
    code_delim = "```json" # 마크다운 코드 블록 구분자

    # 최종 프롬프트 구성
    prompt = f"""
당신은 두 기업('{company_a_name}', '{company_b_name}') 간의 관계를 분석하는 금융 전문가입니다. 제공된 "CONTEXT" 정보만을 사용하여 다음 지침을 엄격히 따르십시오.

## 지침:
1. CONTEXT 내에서 '{company_a_name}'와(과) '{company_b_name}' 사이의 관계를 분석합니다.
2. 다음 세 가지 관계 유형 각각에 대해 조사합니다: {target_relation_types}
3. 각 관계 유형별로, CONTEXT 정보에 기반하여 두 회사 간에 해당 관계가 명시적 또는 암시적으로 존재한다고 판단되면 "connected"를 true로, 그렇지 않으면 false로 설정합니다. ("Ownership"은 직접적인 지분 관계 또는 명확한 계열사 관계를 의미합니다. "Supply"는 제품/서비스/원재료 공급 또는 고객 관계를 포함합니다.)
4. 관계가 존재한다고 판단된 경우(connected가 true), **CONTEXT에서 그 가장 확실한 근거가 되는 문장을 최대 2개**까지 찾아 "evidence" 리스트에 **CONTEXT의 원문 그대로** 포함시키십시오. 근거 문장은 간결하고 명확해야 합니다. 관계가 존재하지 않으면(connected가 false) "evidence"는 반드시 빈 리스트([])여야 합니다.
5. 최종 결과는 반드시 {target_relation_types} 세 가지 유형 모두에 대한 결과를 "relationships" 리스트 안에 포함시켜야 합니다.
6. 출력은 아래 "JSON 스키마" 형식과 정확히 일치하는 **JSON 객체 하나만** 생성해야 합니다. 다른 어떤 텍스트, 설명, 주석, 인사말도 절대 포함하지 마십시오.

## JSON 스키마:
{code_delim}
{json_schema}
{code_delim}

## CONTEXT:
{context_str}

## 출력 JSON:
"""
    return prompt.strip()

# --- 아래 함수들은 예시이며, 실제 구현이 필요할 수 있습니다 ---
# (LLM 호출 및 파싱은 노트북 Cell 6에서 구현하므로 여기서는 생략하거나 주석 처리)

# def extract_relationship_with_llm(prompt: str) -> str | None:
#     # ... LLM 호출 로직 ...
#     logger.warning("extract_relationship_with_llm 함수는 노트북에서 구현/호출됩니다.")
#     return None

# def parse_llm_json_output(llm_response_text: str) -> dict | None:
#     # ... JSON 파싱 로직 ...
#      logger.warning("parse_llm_json_output 함수는 노트북에서 구현/호출됩니다.")
#     return None
