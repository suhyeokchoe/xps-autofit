"""
XPS AutoFit — Analytics Module (v1.1)
==================================================
익명 사용 통계 추적. 두 백엔드 동시 사용:
- PostHog: 시각화 대시보드 (재방문율, funnel 등 자동 분석)
- Google Sheets: raw 데이터 백업 (외부 서비스 의존 최소화)

설계 원칙:
1. 실패해도 앱이 안 죽음 — try/except로 모든 예외 흡수
2. 익명 — 사용자 식별 정보 절대 안 보냄, 익명 session_id만
3. Best effort — 네트워크 느려도 UI 차단 안 함

사용자 식별:
- session_id: Streamlit session 단위 무작위 UUID. 페이지 새로고침 시 갱신.
- 같은 사람이 다음 날 다시 와도 새 ID. 진짜 재방문 추적은 PostHog가
  쿠키로 하지만 우리는 강제 안 함 (privacy 우선).
"""
import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional


# ===================================================================
# 전역 상태 (Streamlit session 안에서 유지)
# ===================================================================
def _get_session_id():
    """현재 Streamlit 세션의 익명 ID. 없으면 생성."""
    try:
        import streamlit as st
        if '_analytics_session_id' not in st.session_state:
            st.session_state['_analytics_session_id'] = str(uuid.uuid4())
        return st.session_state['_analytics_session_id']
    except Exception:
        # Streamlit 없는 환경 (테스트) → 새 ID
        return str(uuid.uuid4())


# ===================================================================
# 설정 로드 (Streamlit secrets 또는 환경변수)
# ===================================================================
def _get_config():
    """
    secrets에서 PostHog와 Google Sheets 설정 로드.
    설정 없으면 None 반환 → 해당 백엔드 비활성화.
    """
    config = {
        'posthog_key': None,
        'posthog_host': 'https://us.i.posthog.com',
        'gsheet_id': None,
        'gsheet_credentials': None,
        'enabled': False,
    }
    try:
        import streamlit as st
        # Streamlit Cloud secrets
        if 'analytics' in st.secrets:
            sec = st.secrets['analytics']
            config['posthog_key'] = sec.get('posthog_key')
            config['posthog_host'] = sec.get('posthog_host', config['posthog_host'])
            config['gsheet_id'] = sec.get('gsheet_id')
            # Google credentials은 JSON 형태 또는 dict로 들어옴
            cred = sec.get('gsheet_credentials')
            if cred:
                if isinstance(cred, str):
                    try:
                        config['gsheet_credentials'] = json.loads(cred)
                    except json.JSONDecodeError:
                        config['gsheet_credentials'] = None
                else:
                    config['gsheet_credentials'] = dict(cred)
    except Exception:
        # secrets 없거나 읽기 실패 — 환경변수 fallback
        config['posthog_key'] = os.environ.get('POSTHOG_KEY')
        config['gsheet_id'] = os.environ.get('GSHEET_ID')
        cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if cred_path and os.path.exists(cred_path):
            try:
                with open(cred_path) as f:
                    config['gsheet_credentials'] = json.load(f)
            except Exception:
                pass

    # 둘 중 하나라도 설정되면 enabled
    config['enabled'] = bool(
        config['posthog_key'] or
        (config['gsheet_id'] and config['gsheet_credentials'])
    )
    return config


# ===================================================================
# PostHog 클라이언트 (lazy)
# ===================================================================
_posthog_client = None

def _get_posthog():
    """PostHog 클라이언트 반환 (lazy init)."""
    global _posthog_client
    if _posthog_client is not None:
        return _posthog_client
    cfg = _get_config()
    if not cfg['posthog_key']:
        return None
    try:
        from posthog import Posthog
        _posthog_client = Posthog(
            project_api_key=cfg['posthog_key'],
            host=cfg['posthog_host'],
        )
        return _posthog_client
    except Exception:
        return None


# ===================================================================
# Google Sheets 클라이언트 (lazy)
# ===================================================================
_gsheet_client = None

def _get_gsheet():
    """Google Sheets 객체(워크시트) 반환 (lazy init)."""
    global _gsheet_client
    if _gsheet_client is not None:
        return _gsheet_client
    cfg = _get_config()
    if not cfg['gsheet_id'] or not cfg['gsheet_credentials']:
        return None
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive',
        ]
        creds = Credentials.from_service_account_info(
            cfg['gsheet_credentials'], scopes=scopes
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(cfg['gsheet_id'])
        _gsheet_client = sh.sheet1  # 첫 번째 시트
        return _gsheet_client
    except Exception:
        return None


# ===================================================================
# Public API: 이벤트 전송
# ===================================================================
def track_event(event_type: str, properties: Optional[dict] = None):
    """
    이벤트를 두 백엔드(PostHog, Google Sheets)에 동시 전송.
    실패는 silent. UI 차단 없음.

    Args:
        event_type: 'app_opened', 'data_uploaded', 'fitting_completed' 등
        properties: 부가 속성 dict
            예: {'mode': 'auto', 'region': 'O1s', 'r_squared': 0.998}
    """
    properties = properties or {}
    session_id = _get_session_id()
    timestamp_iso = datetime.now(timezone.utc).isoformat()

    # PostHog
    try:
        ph = _get_posthog()
        if ph is not None:
            ph.capture(
                distinct_id=session_id,
                event=event_type,
                properties={
                    **properties,
                    '$lib': 'xps-autofit',
                }
            )
    except Exception:
        pass

    # Google Sheets
    try:
        ws = _get_gsheet()
        if ws is not None:
            row = [
                timestamp_iso,
                session_id,
                event_type,
                properties.get('mode', ''),
                properties.get('region', ''),
                properties.get('file_type', ''),
                properties.get('r_squared', ''),
                json.dumps({k: v for k, v in properties.items()
                            if k not in ('mode', 'region', 'file_type', 'r_squared')},
                           ensure_ascii=False),
            ]
            ws.append_row(row, value_input_option='USER_ENTERED')
    except Exception:
        pass


# ===================================================================
# 편의 함수 — 자주 쓰는 이벤트들 wrapper
# ===================================================================
def track_app_opened():
    """앱 진입 시 1회만 호출 (session 단위)."""
    try:
        import streamlit as st
        # 같은 세션에 중복 호출 방지
        if st.session_state.get('_analytics_app_opened'):
            return
        st.session_state['_analytics_app_opened'] = True
    except Exception:
        pass
    track_event('app_opened')


def track_data_uploaded(file_type: str, file_size: int = 0,
                          region: str = 'unknown'):
    """파일 업로드 성공 시."""
    track_event('data_uploaded', {
        'file_type': file_type,
        'file_size': file_size,
        'region': region,
    })


def track_fitting_completed(mode: str, region: str = 'unknown',
                              r_squared: float = 0.0,
                              n_components: int = 0,
                              file_type: str = ''):
    """피팅 완료 시 (자동/Expert/Survey 공통)."""
    track_event('fitting_completed', {
        'mode': mode,             # 'auto' / 'expert' / 'survey'
        'region': region,
        'r_squared': round(r_squared, 4) if r_squared else 0,
        'n_components': n_components,
        'file_type': file_type,
    })


def track_result_downloaded(download_type: str, mode: str = ''):
    """결과 다운로드 시."""
    track_event('result_downloaded', {
        'download_type': download_type,  # 'csv_params' / 'csv_curves' / 'png' 등
        'mode': mode,
    })


def track_error(error_type: str, where: str = '', details: str = ''):
    """에러 발생 시 (스택 트레이스 X — 익명성 보장)."""
    track_event('error_occurred', {
        'error_type': error_type,
        'where': where,
        # 너무 길면 자르기
        'details': details[:200] if details else '',
    })


# ===================================================================
# 헬스체크 — 본인이 디버그용으로 사용 가능
# ===================================================================
def check_status() -> dict:
    """
    분석 백엔드 상태 확인.
    반환값: {'posthog': bool, 'gsheet': bool, 'enabled': bool}
    """
    cfg = _get_config()
    return {
        'posthog_configured': bool(cfg['posthog_key']),
        'gsheet_configured': bool(cfg['gsheet_id'] and cfg['gsheet_credentials']),
        'posthog_client_ok': _get_posthog() is not None,
        'gsheet_client_ok': _get_gsheet() is not None,
        'enabled': cfg['enabled'],
    }
