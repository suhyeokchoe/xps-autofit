"""
XPS Auto-fitting Engine v0.3
================================================
v0.2 대비 추가:
- 스핀-오빗 doublet 자동 처리 (p, d, f orbital)
  · ΔBE와 면적비를 물리 상수로 강제
  · 같은 화학상태는 FWHM/η 공유
- Doublet mode와 singlet mode 자동 분기
- 파라미터 테이블 BE 내림차순 정렬 (XPS 관례)

물리 제약:
  2p: j=3/2,1/2 → area ratio 2:1
  3d: j=5/2,3/2 → area ratio 3:2
  4f: j=7/2,5/2 → area ratio 4:3
"""
import csv
import io
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from pathlib import Path

# -------------------------------------------------------------------
# 원소별 prior
# -------------------------------------------------------------------
ELEMENT_PRIORS = {
    # Singlet: (BE_min, BE_max, FWHM_min, FWHM_max)
    'F1s':   (680, 695, 0.8, 3.5),
    'C1s':   (280, 295, 0.7, 2.5),
    'O1s':   (525, 540, 0.8, 3.0),
    'N1s':   (395, 410, 0.8, 2.5),
    'Li1s':  (50, 62, 0.7, 2.0),
    # Al 2p: 도메인 관행상 doublet split 작아서(0.4 eV) singlet으로 처리
    'Al2p':  (68, 80, 0.7, 2.0),
    # Si 2p: 마찬가지로 split 0.6 eV로 매우 작음 (싱글릿/더블릿 혼용)
    'Si2p_singlet':  (95, 108, 0.6, 2.0),
    # P 2p, S 2p, Cl 2p: doublet이지만 split 작아서 보통 singlet 처리
    'P2p':   (128, 138, 0.8, 2.0),
    'S2p':   (158, 170, 0.8, 2.0),
}

# Doublet: (BE_min, BE_max, FWHM_min, FWHM_max, delta_BE, area_ratio)
# delta_BE: main(낮은 BE) 기준 minor는 +delta_BE
# area_ratio = amp_main / amp_minor (같은 FWHM이면 면적비와 동일)
DOUBLET_PRIORS = {
    # 2p → 2:1
    'Cu2p':  (925, 970, 0.9, 3.0, 19.8,  2.0),
    'Ti2p':  (450, 475, 0.8, 2.5, 5.7,   2.0),
    'Si2p':  (95,  110, 0.6, 2.5, 0.6,   2.0),
    'Fe2p':  (700, 740, 1.0, 4.0, 13.6,  2.0),
    'Ni2p':  (845, 890, 1.0, 3.5, 17.3,  2.0),
    # 3d → 3:2
    'Sn3d':  (480, 500, 0.7, 2.0, 8.41,  1.5),
    'In3d':  (440, 460, 0.7, 2.0, 7.55,  1.5),
    'Mo3d':  (225, 245, 0.8, 2.5, 3.15,  1.5),
    'Ag3d':  (365, 380, 0.7, 2.0, 6.00,  1.5),
    # 4f → 4:3
    'Au4f':  (80,  92,  0.6, 1.8, 3.65,  1.333),
    'W4f':   (30,  42,  0.7, 2.0, 2.15,  1.333),
}


def detect_region(be_min, be_max):
    candidates = []
    for name, tup in ELEMENT_PRIORS.items():
        lo, hi = tup[0], tup[1]
        if be_min >= lo - 5 and be_max <= hi + 5:
            candidates.append(name)
    for name, tup in DOUBLET_PRIORS.items():
        lo, hi = tup[0], tup[1]
        if be_min >= lo - 5 and be_max <= hi + 5:
            candidates.append(name)
    return candidates[0] if candidates else None


def is_doublet(region):
    return region in DOUBLET_PRIORS


# -------------------------------------------------------------------
# CSV 로딩
# -------------------------------------------------------------------
def load_xps_csv(path_or_text, source_name='uploaded'):
    is_path = False
    if isinstance(path_or_text, (str, Path)):
        s = str(path_or_text)
        if len(s) < 500 and '\n' not in s:
            try:
                if Path(s).exists():
                    is_path = True
            except (OSError, ValueError):
                is_path = False

    if is_path:
        rows = list(csv.reader(open(path_or_text, encoding='utf-8-sig')))
        source_name = Path(path_or_text).name
    else:
        text = path_or_text if isinstance(path_or_text, str) else '\n'.join(path_or_text)
        rows = list(csv.reader(io.StringIO(text)))

    meta = {'source_file': source_name, 'region': None}
    for row in rows[:20]:
        line = ','.join(row)
        if 'Scan.VGD' in line or '.VGD' in line:
            for part in line.split('\\'):
                if 'Scan' in part and '.VGD' in part:
                    meta['region'] = part.replace(' Scan.VGD', '').strip().strip(',')
                    break

    be, counts = [], []
    for row in rows:
        if len(row) < 2:
            continue
        try:
            x = float(row[0])
        except (ValueError, IndexError):
            continue
        y = None
        if len(row) >= 3:
            try:
                y = float(row[2])
            except (ValueError, IndexError):
                y = None
        if y is None:
            try:
                y = float(row[1])
            except (ValueError, IndexError):
                continue
        be.append(x); counts.append(y)
    be = np.array(be); counts = np.array(counts)
    if len(be) == 0:
        raise ValueError("데이터 행을 찾지 못했습니다. CSV 구조를 확인하세요.")
    if be[0] > be[-1]:
        be, counts = be[::-1], counts[::-1]

    if meta['region'] is None or meta['region'] == 'unknown':
        meta['region'] = detect_region(be.min(), be.max()) or 'unknown'
    return be, counts, meta


def load_xps_excel(file_obj_or_path, source_name='uploaded',
                     sheet_name=None, counts_col_idx=None):
    """
    Excel(.xlsx, .xls) 파일에서 BE / Counts 데이터 로드.

    자동 휴리스틱 (counts_col_idx=None일 때):
    1) 헤더 행에서 'binding'/'energy' 키워드 → BE 컬럼
    2) 헤더에서 'count'/'cps' 키워드 → Counts 컬럼
    3) 키워드 매칭 안 되면: 첫 숫자 컬럼=BE, 두 번째=Counts
       단, "두 번째"는 BE 다음에 나오는 첫 데이터 컬럼 (raw로 추정)
    4) CasaXPS export 같은 경우 raw counts는 보통 BE 바로 다음 컬럼

    사용자 override (counts_col_idx 지정):
    - 그 컬럼을 그대로 Counts로 사용. BE는 자동 감지된 것 사용.

    meta에 'available_columns' 포함 → app.py가 dropdown UI 만들 수 있음.

    Args:
        file_obj_or_path: 파일 경로 또는 파일류 객체
        source_name: 파일명
        sheet_name: 시트 이름 (None이면 첫 시트)
        counts_col_idx: Counts 컬럼 인덱스 강제 지정 (None이면 자동)

    Returns:
        (be, counts, meta)
    """
    import pandas as pd

    # Pandas로 읽기
    try:
        if sheet_name is None:
            df = pd.read_excel(file_obj_or_path, sheet_name=0, header=None)
        else:
            df = pd.read_excel(file_obj_or_path, sheet_name=sheet_name, header=None)
    except ImportError as e:
        if 'xlrd' in str(e).lower():
            raise ImportError(
                ".xls 파일을 읽으려면 xlrd 라이브러리가 필요합니다. "
                "requirements.txt에 'xlrd==2.0.1' 추가하거나, "
                "Excel에서 .xlsx로 다시 저장하세요."
            )
        raise
    except Exception as e:
        raise ValueError(f"Excel 파일 읽기 실패: {e}")

    if df.empty:
        raise ValueError("Excel 시트가 비어있습니다.")

    # 헤더 행 찾기 — 'binding energy' 같은 명확한 패턴 우선
    header_row = None

    # Pass 1: 'binding energy' 또는 'binding'이 단독으로 들어간 행 (가장 명확)
    for row_idx in range(min(25, len(df))):
        row_vals = [str(v).lower() for v in df.iloc[row_idx].dropna()]
        for v in row_vals:
            # 'binding energy' 같은 명확한 헤더 패턴
            if 'binding energy' in v or v.strip() == 'binding' or 'b.e.' in v:
                header_row = row_idx
                break
        if header_row is not None:
            break

    # Pass 2 (fallback): 'energy' 단독 (덜 명확, 메타정보일 수도)
    if header_row is None:
        for row_idx in range(min(25, len(df))):
            row_str = ' '.join([str(v).lower() for v in df.iloc[row_idx].dropna()])
            if 'energy' in row_str:
                header_row = row_idx
                break

    # BE 컬럼 식별
    be_col_idx = None
    if header_row is not None:
        header_vals = [str(v).lower() if v is not None else '' for v in df.iloc[header_row]]
        for i, h in enumerate(header_vals):
            if any(k in h for k in ['binding', 'energy', 'b.e.']):
                be_col_idx = i
                break
        data_start = header_row + 1
        # CasaXPS는 헤더 다음 단위 행(eV, Counts/s 등)이 있음 — 건너뛰기
        if data_start < len(df):
            try_units = df.iloc[data_start]
            unit_str = ' '.join([str(v).lower() for v in try_units.dropna()])
            if 'ev' in unit_str or 'count' in unit_str or 'cps' in unit_str:
                data_start += 1
    else:
        # fallback: 첫 숫자 행 찾기
        data_start = None
        for row_idx in range(min(30, len(df))):
            row = df.iloc[row_idx]
            numeric_count = sum(
                1 for v in row.dropna()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
            if numeric_count >= 2:
                data_start = row_idx
                break
        if data_start is None:
            raise ValueError("데이터로 보이는 숫자 행을 찾지 못했습니다.")

    # 데이터 영역에서 숫자형 컬럼 찾기
    data_df = df.iloc[data_start:].apply(pd.to_numeric, errors='coerce')
    numeric_cols = [
        i for i in range(data_df.shape[1])
        if data_df.iloc[:, i].notna().sum() > len(data_df) * 0.5
    ]

    if len(numeric_cols) < 2:
        raise ValueError(
            f"BE/Counts로 쓸 숫자 컬럼이 충분하지 않습니다. "
            f"발견된 숫자 컬럼: {len(numeric_cols)}개"
        )

    # BE 컬럼 결정
    if be_col_idx is None or be_col_idx not in numeric_cols:
        be_col_idx = numeric_cols[0]

    # Counts 컬럼 결정
    candidates = [c for c in numeric_cols if c != be_col_idx]
    if not candidates:
        raise ValueError("BE 외에 다른 숫자 컬럼이 없습니다.")

    if counts_col_idx is not None:
        # 사용자 지정 우선
        if counts_col_idx not in candidates:
            raise ValueError(
                f"지정한 Counts 컬럼(index={counts_col_idx})이 데이터에 없습니다. "
                f"사용 가능한 컬럼: {candidates}"
            )
        chosen_counts_col = counts_col_idx
    else:
        # 자동: BE 바로 다음의 데이터 컬럼 (CasaXPS export 관행)
        # numeric_cols 중 be_col_idx 다음에 오는 첫 번째
        chosen_counts_col = candidates[0]
        for c in candidates:
            if c > be_col_idx:
                chosen_counts_col = c
                break

    # 컬럼 후보 정보 (사용자 dropdown UI 위해 — 컬럼명 추출)
    available_columns = []
    if header_row is not None:
        header_vals = df.iloc[header_row].tolist()
        for c in candidates:
            name = str(header_vals[c]) if c < len(header_vals) and pd.notna(header_vals[c]) else f'Column {c}'
            available_columns.append({
                'index': int(c),
                'name': name.strip(),
                'is_default': (c == chosen_counts_col),
            })
    else:
        for c in candidates:
            available_columns.append({
                'index': int(c),
                'name': f'Column {c}',
                'is_default': (c == chosen_counts_col),
            })

    # 데이터 추출
    raw_be = pd.to_numeric(df.iloc[data_start:, be_col_idx], errors='coerce')
    raw_counts = pd.to_numeric(df.iloc[data_start:, chosen_counts_col], errors='coerce')

    valid = raw_be.notna() & raw_counts.notna()
    be = raw_be[valid].values.astype(float)
    counts = raw_counts[valid].values.astype(float)

    if len(be) == 0:
        raise ValueError("Excel에서 유효한 BE/Counts 데이터를 추출하지 못했습니다.")

    # BE 정렬 (오름차순)
    if be[0] > be[-1]:
        be = be[::-1]; counts = counts[::-1]

    # 선택된 컬럼 이름 (메타용)
    chosen_name = next(
        (a['name'] for a in available_columns if a['index'] == chosen_counts_col),
        f'Column {chosen_counts_col}'
    )

    meta = {
        'source_file': source_name,
        'region': None,
        'excel_sheet': sheet_name or 'first sheet',
        'detected_columns': f"BE=col{be_col_idx}, Counts='{chosen_name}' (col{chosen_counts_col})",
        'be_col_idx': int(be_col_idx),
        'counts_col_idx': int(chosen_counts_col),
        'available_columns': available_columns,  # app.py가 dropdown 만들 때 사용
    }

    if meta['region'] is None or meta['region'] == 'unknown':
        meta['region'] = detect_region(be.min(), be.max()) or 'unknown'

    return be, counts, meta


def list_excel_sheets(file_obj_or_path):
    """
    Excel 파일의 시트 목록을 반환.
    각 시트에 데이터가 있는지 여부도 함께 표시.

    Args:
        file_obj_or_path: 파일 경로 또는 파일류 객체

    Returns:
        list of dict: [{'name': str, 'has_data': bool, 'n_rows': int, 'preview': str}, ...]
    """
    import pandas as pd

    try:
        # ExcelFile은 시트 목록을 빠르게 가져옴
        xl = pd.ExcelFile(file_obj_or_path)
        sheet_names = xl.sheet_names
    except ImportError as e:
        if 'xlrd' in str(e).lower():
            raise ImportError(
                ".xls 파일을 읽으려면 xlrd 라이브러리가 필요합니다. "
                "Excel에서 .xlsx로 다시 저장하거나, 관리자에게 문의하세요."
            )
        raise
    except Exception as e:
        raise ValueError(f"Excel 파일 시트 목록 조회 실패: {e}")

    sheets_info = []
    for name in sheet_names:
        try:
            df = pd.read_excel(file_obj_or_path, sheet_name=name, header=None,
                                nrows=30)  # 처음 30행만 미리보기
            has_data = False
            n_numeric_rows = 0
            for _, row in df.iterrows():
                vals = row.dropna()
                if len(vals) >= 2:
                    has_data = True
                    # 숫자 행 카운트
                    try:
                        nums = sum(1 for v in vals if isinstance(v, (int, float)))
                        if nums >= 2:
                            n_numeric_rows += 1
                    except Exception:
                        pass

            # 미리보기 텍스트
            preview = ''
            for _, row in df.iterrows():
                vals = [str(v) for v in row.dropna()][:3]
                if vals:
                    preview = ' | '.join(vals)
                    break

            sheets_info.append({
                'name': name,
                'has_data': has_data and n_numeric_rows >= 5,
                'n_rows': len(df),
                'preview': preview[:80],
            })
        except Exception:
            sheets_info.append({
                'name': name,
                'has_data': False,
                'n_rows': 0,
                'preview': '(읽기 실패)',
            })

    return sheets_info


def load_xps_data(file_obj_or_path, source_name='uploaded',
                    sheet_name=None, counts_col_idx=None):
    """
    통합 entry point: 파일 확장자 + 객체 타입에 따라 적절한 로더 자동 선택.

    분기 로직:
    1) sheet_name이 명시되었으면 → 무조건 Excel
    2) 확장자가 .xlsx/.xls 이면 → Excel
    3) source_name이나 경로에 .xlsx/.xls가 포함되어 있으면 → Excel
    4) 그 외 → CSV/TXT

    Args:
        file_obj_or_path: 경로(str) 또는 파일류 객체 (BytesIO/UploadedFile) 또는 텍스트
        source_name: 파일명 (확장자 추출용)
        sheet_name: Excel 시트 이름 (지정 시 Excel로 강제 분기)
        counts_col_idx: Excel Counts 컬럼 인덱스 강제 지정

    Returns:
        (be, counts, meta)
    """
    # 1) sheet_name이 있으면 Excel 확정
    if sheet_name is not None:
        return load_xps_excel(file_obj_or_path,
                                source_name=source_name,
                                sheet_name=sheet_name,
                                counts_col_idx=counts_col_idx)

    # 2) 확장자 검사 (source_name 또는 경로)
    fname = source_name or ''
    if isinstance(file_obj_or_path, (str, Path)):
        try:
            p = Path(str(file_obj_or_path))
            if p.exists() or p.suffix:
                fname = p.name
        except (OSError, ValueError):
            pass

    fname_lower = fname.lower()
    # ".xlsx"가 어디든 포함되어 있으면 Excel (예: "file.xlsx :: Sheet1")
    is_excel = '.xlsx' in fname_lower or '.xls ' in fname_lower or fname_lower.endswith('.xls')

    if is_excel:
        return load_xps_excel(file_obj_or_path,
                                source_name=source_name,
                                sheet_name=sheet_name,
                                counts_col_idx=counts_col_idx)

    # 3) bytes/BytesIO인데 Excel 마커가 없으면 위험 — 명시적 에러
    if hasattr(file_obj_or_path, 'read') and not isinstance(file_obj_or_path, (str, Path)):
        # 파일 객체인데 확장자를 모르면 CSV/TXT로 가정 (텍스트 디코딩 시도)
        # 단, BytesIO면 텍스트로 변환 필요
        try:
            data = file_obj_or_path.read()
            if isinstance(data, bytes):
                text = data.decode('utf-8-sig', errors='replace')
                return load_xps_csv(text, source_name=source_name)
            else:
                return load_xps_csv(data, source_name=source_name)
        except Exception as e:
            raise ValueError(
                f"파일 형식을 인식할 수 없습니다 (source_name={source_name}). "
                f"파일명에 .csv, .txt, .xlsx, .xls 확장자가 있어야 합니다. "
                f"내부 에러: {e}"
            )

    # 4) 그 외: CSV 또는 텍스트 (기존 동작)
    return load_xps_csv(file_obj_or_path, source_name=source_name)


# -------------------------------------------------------------------
# Shirley background
# -------------------------------------------------------------------
def shirley_background(x, y, max_iter=60, tol=1e-6,
                        anchor_left=None, anchor_right=None,
                        auto_anchor=True):
    """
    Iterative Shirley background.

    Parameters:
        x, y: data (x must be sorted ascending in BE)
        max_iter, tol: 수렴 파라미터
        anchor_left, anchor_right: 사용자 지정 anchor BE 값.
                                    None이면 자동 감지 사용.
        auto_anchor: True면 anchor 자동 감지. False면 양 끝점 사용.

    표준 Shirley 정의:
    - BG[0]  = y[0]  (낮은 BE 끝)
    - BG[-1] = y[-1] (높은 BE 끝)
    - 중간은 cumulative integral로 결정

    Anchor 모드 (auto_anchor=True 또는 anchor_left/right 지정):
    - 피크가 없는 양 끝 영역(anchor)을 자동 감지
    - Shirley는 anchor 사이에서만 계산
    - Anchor 바깥은 raw data를 따라감 (BG = data)
    """
    # Anchor 자동 감지 또는 사용자 지정
    if anchor_left is not None or anchor_right is not None:
        # 사용자 지정 (BE 값 → index 변환)
        if anchor_left is not None:
            left_idx = int(np.argmin(np.abs(x - anchor_left)))
        else:
            left_idx = 0
        if anchor_right is not None:
            right_idx = int(np.argmin(np.abs(x - anchor_right)))
        else:
            right_idx = len(y) - 1
        if left_idx > right_idx:
            left_idx, right_idx = right_idx, left_idx
        use_anchor = True
    elif auto_anchor:
        left_idx, right_idx = _detect_anchor_indices(x, y)
        use_anchor = True
    else:
        left_idx, right_idx = 0, len(y) - 1
        use_anchor = False

    # Shirley는 [left_idx, right_idx] 구간에서만 계산
    if use_anchor and (left_idx > 0 or right_idx < len(y) - 1):
        x_mid = x[left_idx:right_idx+1]
        y_mid = y[left_idx:right_idx+1]
        bg_mid = _shirley_iterate(x_mid, y_mid, max_iter, tol)
        # 전체 BG: 바깥은 raw data
        bg_full = np.zeros_like(y, dtype=float)
        bg_full[:left_idx] = y[:left_idx]
        bg_full[left_idx:right_idx+1] = bg_mid
        bg_full[right_idx+1:] = y[right_idx+1:]
        return bg_full
    else:
        return _shirley_iterate(x, y, max_iter, tol)


def _shirley_iterate(x, y, max_iter=60, tol=1e-6):
    """Core Shirley iteration on full given range."""
    B = np.zeros_like(y, dtype=float)
    I_start = y[0]
    I_end = y[-1]
    for _ in range(max_iter):
        cum = np.zeros_like(y, dtype=float)
        for i in range(1, len(y)):
            cum[i] = cum[i-1] + 0.5 * (
                (y[i-1] - I_start - B[i-1]) + (y[i] - I_start - B[i])
            ) * (x[i] - x[i-1])
        denom = cum[-1] if abs(cum[-1]) > 1e-12 else 1.0
        k = (I_end - I_start) / denom
        B_new = k * cum
        if np.max(np.abs(B_new - B)) < tol * max(abs(I_end - I_start), 1.0):
            B = B_new; break
        B = B_new
    return I_start + B


def _detect_anchor_indices(x, y, window_eV=1.5, threshold_ratio=0.08):
    """
    피크 영역 자동 감지로 anchor index 결정.

    Algorithm:
    - 1차 미분 계산
    - 양 끝에서 출발, |dy/dx|가 처음 threshold 넘는 곳 = anchor 끝
    - 양 끝 anchor 영역에서는 BG = raw data (변동 없음)
    - 가운데 = 피크 영역 = Shirley 계산 영역
    """
    step = abs(x[1] - x[0])
    win = max(7, int(window_eV / step))
    if win % 2 == 0: win += 1
    if win >= len(y):
        win = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2

    try:
        dy = savgol_filter(y, win, 3, deriv=1)
    except Exception:
        return 0, len(y) - 1

    abs_dy = np.abs(dy)
    threshold = abs_dy.max() * threshold_ratio

    # 왼쪽: 인덱스 0부터 출발해서 첫 threshold 초과 지점 직전
    left_idx = 0
    for i in range(len(abs_dy)):
        if abs_dy[i] > threshold:
            left_idx = max(0, i - 1)
            break

    # 오른쪽: 끝에서 거꾸로
    right_idx = len(abs_dy) - 1
    for i in range(len(abs_dy) - 1, -1, -1):
        if abs_dy[i] > threshold:
            right_idx = min(len(y) - 1, i + 1)
            break

    # 안전 마진: 각 끝에서 최소 5 포인트는 anchor
    left_idx = max(left_idx, 5)
    right_idx = min(right_idx, len(y) - 6)

    # left가 right보다 크거나 너무 가까우면 fallback (전체 영역 사용)
    if right_idx - left_idx < len(y) // 4:
        left_idx, right_idx = 0, len(y) - 1

    return left_idx, right_idx


# -------------------------------------------------------------------
# Pseudo-Voigt
# -------------------------------------------------------------------
def pseudo_voigt(x, amp, center, fwhm, eta):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm / 2
    G = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    L = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return amp * ((1 - eta) * G + eta * L)


def multi_pv(x, *params):
    n = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n):
        y = y + pseudo_voigt(x, *params[i*4:i*4+4])
    return y


def multi_doublet_pv(x, delta_BE, area_ratio, *params):
    """
    n_states 화학상태, 각각 doublet.
    params per state: [amp_main, center_main, fwhm_shared, eta_shared]
    제약: center_minor = center_main + delta_BE
          amp_minor    = amp_main / area_ratio
          fwhm, eta 동일 (같은 화학상태)
    """
    n_states = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n_states):
        amp_m, c_m, fwhm, eta = params[i*4:i*4+4]
        y = y + pseudo_voigt(x, amp_m, c_m, fwhm, eta)
        amp_n = amp_m / area_ratio
        c_n = c_m + delta_BE
        y = y + pseudo_voigt(x, amp_n, c_n, fwhm, eta)
    return y


# -------------------------------------------------------------------
# 피크 감지
# -------------------------------------------------------------------
def detect_peaks_v2(x, y_corr, region=None):
    win = max(7, len(y_corr) // 25)
    if win % 2 == 0: win += 1
    y_smooth = savgol_filter(y_corr, win, 3)
    d2 = savgol_filter(y_corr, win, 3, deriv=2)

    prom = max(y_smooth) * 0.05
    main_idx, _ = find_peaks(y_smooth, prominence=prom,
                              distance=max(5, len(x)//40))
    neg_d2 = -d2
    main_d2 = [neg_d2[i] for i in main_idx] if len(main_idx) else [np.max(neg_d2)]
    d2_thr = np.median(main_d2) * 0.30
    shoulder_idx, _ = find_peaks(neg_d2, prominence=d2_thr,
                                  distance=max(5, len(x)//40))

    all_idx = sorted(set(list(main_idx) + list(shoulder_idx)))
    merged = []
    merge_tol = 0.7
    for idx in all_idx:
        if merged and abs(x[idx] - x[merged[-1]]) < merge_tol:
            if y_smooth[idx] > y_smooth[merged[-1]]:
                merged[-1] = idx
        else:
            merged.append(idx)
    edge = len(x) // 20
    merged = [i for i in merged if edge < i < len(x) - edge]
    return np.array(merged, dtype=int), y_smooth


# -------------------------------------------------------------------
# Singlet 피팅
# -------------------------------------------------------------------
def fit_n_peaks(x, y_corr, n, init_centers, region=None):
    if region in ELEMENT_PRIORS:
        _, _, fwhm_min, fwhm_max = ELEMENT_PRIORS[region]
    elif region in DOUBLET_PRIORS:
        _, _, fwhm_min, fwhm_max, _, _ = DOUBLET_PRIORS[region]
    else:
        fwhm_min, fwhm_max = 0.5, 5.0

    p0, lo, hi = [], [], []
    for c0 in init_centers[:n]:
        idx0 = int(np.argmin(np.abs(x - c0)))
        amp0 = max(y_corr[idx0], max(y_corr) * 0.05)
        p0 += [amp0, c0, np.mean([fwhm_min, fwhm_max]), 0.3]
        lo += [amp0 * 0.05, c0 - 1.5, fwhm_min, 0.0]
        hi += [amp0 * 5.0,  c0 + 1.5, fwhm_max, 1.0]

    try:
        popt, _ = curve_fit(multi_pv, x, y_corr,
                            p0=p0, bounds=(lo, hi), maxfev=15000)
        y_fit = multi_pv(x, *popt)
        resid = y_corr - y_fit
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y_corr - np.mean(y_corr)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rms = float(np.sqrt(ss_res / len(x)))
        N = len(x); k = 4 * n
        aic = N * np.log(ss_res / N + 1e-20) + 2 * k
        return {'popt': popt, 'y_fit': y_fit, 'r2': r2, 'rms': rms,
                'aic': aic, 'n_peaks': n, 'mode': 'singlet'}
    except Exception:
        return None


# -------------------------------------------------------------------
# Doublet 피팅
# -------------------------------------------------------------------
def fit_n_doublets(x, y_corr, n_states, init_centers_main, region):
    _, _, fwhm_min, fwhm_max, delta_BE, area_ratio = DOUBLET_PRIORS[region]

    p0, lo, hi = [], [], []
    for c0 in init_centers_main[:n_states]:
        idx0 = int(np.argmin(np.abs(x - c0)))
        amp0 = max(y_corr[idx0], max(y_corr) * 0.05)
        p0 += [amp0, c0, np.mean([fwhm_min, fwhm_max]), 0.3]
        lo += [amp0 * 0.05, c0 - 2.0, fwhm_min, 0.0]
        hi += [amp0 * 5.0,  c0 + 2.0, fwhm_max, 1.0]

    def model(x_arr, *params):
        return multi_doublet_pv(x_arr, delta_BE, area_ratio, *params)

    try:
        popt, _ = curve_fit(model, x, y_corr,
                            p0=p0, bounds=(lo, hi), maxfev=15000)
        y_fit = model(x, *popt)
        resid = y_corr - y_fit
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y_corr - np.mean(y_corr)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rms = float(np.sqrt(ss_res / len(x)))
        N = len(x); k = 4 * n_states  # 자유도: state당 4
        aic = N * np.log(ss_res / N + 1e-20) + 2 * k
        return {'popt': popt, 'y_fit': y_fit, 'r2': r2, 'rms': rms,
                'aic': aic, 'n_peaks': n_states, 'mode': 'doublet',
                'delta_BE': delta_BE, 'area_ratio': area_ratio}
    except Exception:
        return None


# -------------------------------------------------------------------
# 자동 파이프라인 (singlet/doublet 자동 분기)
# -------------------------------------------------------------------
def auto_fit_v3(be, counts, meta=None, max_peaks=4, bg_kwargs=None):
    """
    자동 피팅. BE 범위가 500 eV 이상이면 자동으로 Survey 모드로 분기.

    bg_kwargs: shirley_background()에 전달할 추가 옵션 dict
               (예: {'auto_anchor': False} or {'anchor_left': 535, 'anchor_right': 528})
    """
    bg_kwargs = bg_kwargs or {}

    # ---- 자동 분기: Survey vs Narrow ----
    try:
        from xps_survey import is_survey_scan, analyze_survey
        if is_survey_scan(be):
            survey_result = analyze_survey(be, counts)
            survey_result['meta'] = meta or {}
            return survey_result
    except ImportError:
        pass

    # ---- Narrow 모드 (기존 로직) ----
    meta = meta or {}
    region = meta.get('region')

    bg = shirley_background(be, counts, **bg_kwargs)
    y_corr = counts - bg

    peaks_idx, y_smooth = detect_peaks_v2(be, y_corr, region)
    if len(peaks_idx) == 0:
        return {'success': False, 'reason': 'No peaks detected',
                'be': be, 'counts': counts, 'background': bg}

    init_centers = [float(be[i]) for i in peaks_idx]

    trials = []

    if is_doublet(region):
        # Doublet 우선 시도
        max_states = max(1, min(max_peaks, max(1, len(init_centers) // 2 + 1)))
        sorted_by_be = sorted(init_centers)
        # main 후보: 모든 감지 피크 (fit_n_doublets가 알아서 상위 n개 사용)
        for n in range(1, max_states + 1):
            ranked = sorted(init_centers,
                            key=lambda c: -y_corr[int(np.argmin(np.abs(be - c)))])
            centers = sorted(ranked[:n])
            result = fit_n_doublets(be, y_corr, n, centers, region)
            if result is not None:
                trials.append(result)

        # Doublet이 안 맞으면 singlet fallback
        if not trials or (trials and min(t['r2'] for t in trials) < 0.9):
            for n in range(1, min(max_peaks, len(init_centers)) + 1):
                ranked = sorted(init_centers,
                                key=lambda c: -y_corr[int(np.argmin(np.abs(be - c)))])
                centers = sorted(ranked[:n])
                r = fit_n_peaks(be, y_corr, n, centers, region)
                if r is not None: trials.append(r)
    else:
        # Singlet only
        max_try = min(max_peaks, len(init_centers))
        max_try = max(max_try, 1)
        for n in range(1, max_try + 1):
            ranked = sorted(init_centers,
                            key=lambda c: -y_corr[int(np.argmin(np.abs(be - c)))])
            centers = sorted(ranked[:n])
            result = fit_n_peaks(be, y_corr, n, centers, region)
            if result is not None:
                trials.append(result)

    if not trials:
        return {'success': False, 'reason': 'All fits failed',
                'be': be, 'counts': counts, 'background': bg}

    # AIC 최소 선택 (doublet이 같은 AIC면 우선)
    best = min(trials, key=lambda r: (r['aic'],
                                       0 if r['mode'] == 'doublet' else 1))
    if best['r2'] < 0.9:
        return {'success': False, 'reason': f'Best R²={best["r2"]:.3f} too low',
                'be': be, 'counts': counts, 'background': bg}

    # 컴포넌트 정리
    components = []
    if best['mode'] == 'doublet':
        delta_BE = best['delta_BE']
        area_ratio = best['area_ratio']
        for i in range(best['n_peaks']):
            amp_m, c_m, fwhm, eta = best['popt'][i*4:i*4+4]
            comp_y_m = pseudo_voigt(be, amp_m, c_m, fwhm, eta)
            components.append({
                'amplitude': float(amp_m), 'position': float(c_m),
                'fwhm': float(fwhm), 'eta': float(eta),
                'area': float(abs(np.trapezoid(comp_y_m, be))),
                'curve': comp_y_m,
                'label': f'State {i+1} (main)',
            })
            amp_n = amp_m / area_ratio
            c_n = c_m + delta_BE
            comp_y_n = pseudo_voigt(be, amp_n, c_n, fwhm, eta)
            components.append({
                'amplitude': float(amp_n), 'position': float(c_n),
                'fwhm': float(fwhm), 'eta': float(eta),
                'area': float(abs(np.trapezoid(comp_y_n, be))),
                'curve': comp_y_n,
                'label': f'State {i+1} (minor)',
            })
    else:
        for i in range(best['n_peaks']):
            a, c, f, e = best['popt'][i*4:i*4+4]
            comp_y = pseudo_voigt(be, a, c, f, e)
            components.append({
                'amplitude': float(a), 'position': float(c),
                'fwhm': float(f), 'eta': float(e),
                'area': float(abs(np.trapezoid(comp_y, be))),
                'curve': comp_y,
                'label': f'Peak {i+1}',
            })

    # XPS 관례: 큰 BE → 작은 BE 순
    components.sort(key=lambda c: -c['position'])
    # 정렬 후 Peak 번호 재할당 (Peak 1 = 가장 큰 BE)
    for i, c in enumerate(components):
        if c['label'].startswith('Peak '):
            c['label'] = f'Peak {i+1}'
    total_area = sum(c['area'] for c in components) or 1
    for c in components:
        c['area_pct'] = 100 * c['area'] / total_area

    trial_summary = [{'n_peaks': t['n_peaks'], 'r2': t['r2'],
                      'rms': t['rms'], 'aic': t['aic'],
                      'mode': t['mode']} for t in trials]

    return {
        'success': True, 'meta': meta,
        'region': region,
        'mode': best['mode'],
        'be': be, 'counts': counts, 'background': bg,
        'y_corrected': y_corr, 'y_fit': best['y_fit'],
        'components': components,
        'r_squared': best['r2'], 'rms': best['rms'], 'aic': best['aic'],
        'n_peaks': best['n_peaks'],
        'trials': trial_summary,
        'doublet_info': {
            'delta_BE': best.get('delta_BE'),
            'area_ratio': best.get('area_ratio'),
        } if best['mode'] == 'doublet' else None,
    }


# 하위 호환
auto_fit_v2 = auto_fit_v3


def calibrate_shift(be, shift):
    return be + shift
