import os
import dart_fss as dart
from tenacity import retry, stop_after_attempt, wait_fixed
from config import RETRY_ATTEMPTS, RETRY_WAIT_SECONDS, TARGET_REPORT_CODES, logger

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_fixed(RETRY_WAIT_SECONDS))
def find_latest_annual_report(corp_code, year, report_codes=TARGET_REPORT_CODES):
    try:
        reports = dart.filings.search(
            corp_code=corp_code,
            bgn_de=f"{year}0101",
            end_de=f"{year}1231",
            pblntf_ty='A',
            pblntf_detail_ty=report_codes,
            sort='date',
            sort_mth='desc',
            page_count=1
        )
        if reports and reports.total_count > 0:
            latest_report = reports.page
            logger.info(f"{corp_code} ({year}년) 최신 보고서: {latest_report.report_nm}, 접수번호: {latest_report.rcept_no}")
            return latest_report.rcept_no
        else:
            logger.warning(f"{corp_code} ({year}년) 보고서를 찾을 수 없습니다.")
            return None
    except Exception as e:
        logger.error(f"{corp_code} 보고서 검색 오류: {e}")
        raise

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_fixed(RETRY_WAIT_SECONDS))
def download_report_file(rcept_no, download_path):
    try:
        os.makedirs(download_path, exist_ok=True)
        file_path = dart.filings.download(rcept_no, path=download_path)
        if file_path and os.path.exists(file_path):
            logger.info(f"보고서 {rcept_no} 다운로드 완료: {file_path}")
            return file_path
        else:
            logger.error(f"보고서 {rcept_no} 다운로드 실패")
            return None
    except Exception as e:
        logger.error(f"다운로드 오류 {rcept_no}: {e}")
        raise

def extract_html_from_zip(zip_file_path):
    import zipfile
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            extract_dir = os.path.splitext(zip_file_path)[0]
            os.makedirs(extract_dir, exist_ok=True)
            zip_ref.extractall(extract_dir)
            html_file = None
            for fname in zip_ref.namelist():
                if fname.lower().endswith(('.html', '.htm')):
                    html_file = os.path.join(extract_dir, fname)
                    break
            if html_file and os.path.exists(html_file):
                with open(html_file, 'rb') as f:
                    html_bytes = f.read()
                logger.info(f"HTML 추출 성공: {html_file}")
                return html_bytes
            else:
                logger.error("HTML 파일을 찾을 수 없습니다.")
                return None
    except Exception as e:
        logger.error(f"ZIP 처리 오류: {e}")
        return None

def decode_html(html_bytes):
    from bs4 import BeautifulSoup
    try:
        soup = BeautifulSoup(html_bytes, 'lxml')
        if soup.original_encoding:
            try:
                return html_bytes.decode(soup.original_encoding, errors='replace')
            except Exception:
                return html_bytes.decode('utf-8', errors='replace')
        else:
            return html_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        logger.error(f"HTML 디코딩 오류: {e}")
        return ""
