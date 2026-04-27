"""
XPS Survey Analysis Module (v0.5)
==================================================
Survey scan 자동 분석:
- 원소 자동 식별 (다중 피크 검증으로 가짜 양성 차단)
- atomic % 정량화 (Scofield RSF 기반)

데이터 출처:
- Binding energies: NIST X-ray Photoelectron Spectroscopy Database
- Sensitivity factors: Scofield (1976) photoionization cross-sections,
  with Al Kα (1486.6 eV) excitation
- Auger lines: Wagner et al., Handbook of Auger Electron Spectroscopy

다중 피크 검증 원리:
- 원소 X가 진짜 있으면 Survey에 X의 여러 코어 레벨이 동시에 보여야 함
- 주 피크(가장 강함) + 부 피크 1개 이상 매칭 → 검출 확정
- 부 피크 없이 주 피크만 → "약한 검출" (가짜 양성 가능)
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from dataclasses import dataclass
from typing import List, Optional


# ===================================================================
# 원소 데이터베이스
# ===================================================================
# 각 원소: 코어 레벨 + 결합에너지(eV) + Scofield RSF + 종류(primary/secondary/auger)
# RSF는 Al Kα 기준 (다른 X-ray source는 보정 필요하지만, 대부분 Al Kα 사용)
# RSF 값은 CasaXPS/Avantage에서 사용하는 표준값 사용

ELEMENT_DB = {
    # ---------- Light elements ----------
    'Li': {
        'lines': [
            {'name': '1s', 'be': 54.7, 'rsf': 0.057, 'role': 'primary'},
        ],
    },
    'B': {
        'lines': [
            {'name': '1s', 'be': 188.0, 'rsf': 0.486, 'role': 'primary'},
        ],
    },
    'C': {
        'lines': [
            {'name': '1s',     'be': 284.8, 'rsf': 1.000, 'role': 'primary'},
            {'name': 'KLL',    'be': 1223,  'rsf': 0,     'role': 'auger'},
        ],
    },
    'N': {
        'lines': [
            {'name': '1s',     'be': 399.5, 'rsf': 1.800, 'role': 'primary'},
            {'name': 'KLL',    'be': 1106,  'rsf': 0,     'role': 'auger'},
        ],
    },
    'O': {
        'lines': [
            {'name': '1s',     'be': 530.5, 'rsf': 2.930, 'role': 'primary'},
            {'name': '2s',     'be': 23.0,  'rsf': 0.131, 'role': 'secondary'},
            {'name': 'KLL',    'be': 745,   'rsf': 0,     'role': 'auger'},
        ],
    },
    'F': {
        'lines': [
            {'name': '1s',     'be': 686.5, 'rsf': 4.430, 'role': 'primary'},
            {'name': '2s',     'be': 30.5,  'rsf': 0.142, 'role': 'secondary'},
            {'name': 'KLL',    'be': 832,   'rsf': 0,     'role': 'auger'},
        ],
    },
    'Na': {
        'lines': [
            {'name': '1s',     'be': 1071.0, 'rsf': 8.520,'role': 'primary'},
            {'name': '2s',     'be': 63.0,   'rsf': 0.286,'role': 'secondary'},
            {'name': 'KLL',    'be': 990,    'rsf': 0,    'role': 'auger'},
        ],
    },

    # ---------- 3rd row ----------
    'Mg': {
        'lines': [
            {'name': '1s',  'be': 1303.0, 'rsf': 12.380, 'role': 'primary'},
            {'name': '2s',  'be': 89.0,   'rsf': 0.490,  'role': 'secondary'},
            {'name': '2p',  'be': 49.5,   'rsf': 0.510,  'role': 'secondary'},
        ],
    },
    'Al': {
        'lines': [
            {'name': '2p',  'be': 74.5,   'rsf': 0.537,  'role': 'primary'},
            {'name': '2s',  'be': 118.5,  'rsf': 0.753,  'role': 'secondary'},
        ],
    },
    'Si': {
        'lines': [
            {'name': '2p',  'be': 99.5,   'rsf': 0.817,  'role': 'primary'},
            {'name': '2s',  'be': 150.5,  'rsf': 0.955,  'role': 'secondary'},
        ],
    },
    'P': {
        'lines': [
            {'name': '2p',  'be': 130.0,  'rsf': 1.192,  'role': 'primary'},
            {'name': '2s',  'be': 189.0,  'rsf': 1.190,  'role': 'secondary'},
        ],
    },
    'S': {
        'lines': [
            {'name': '2p',  'be': 164.0,  'rsf': 1.668,  'role': 'primary'},
            {'name': '2s',  'be': 228.0,  'rsf': 1.430,  'role': 'secondary'},
        ],
    },
    'Cl': {
        'lines': [
            {'name': '2p',  'be': 198.5,  'rsf': 2.285,  'role': 'primary'},
            {'name': '2s',  'be': 270.0,  'rsf': 1.690,  'role': 'secondary'},
        ],
    },
    'K': {
        'lines': [
            {'name': '2p',  'be': 293.0,  'rsf': 3.026,  'role': 'primary'},
            {'name': '2s',  'be': 378.0,  'rsf': 1.940,  'role': 'secondary'},
        ],
    },
    'Ca': {
        'lines': [
            {'name': '2p',  'be': 347.0,  'rsf': 3.893,  'role': 'primary'},
            {'name': '2s',  'be': 438.0,  'rsf': 2.250,  'role': 'secondary'},
        ],
    },

    # ---------- Transition metals (3d) ----------
    'Ti': {
        'lines': [
            {'name': '2p3/2','be': 458.5, 'rsf': 7.910,  'role': 'primary'},
            {'name': '2p1/2','be': 464.2, 'rsf': 4.000,  'role': 'secondary'},
            {'name': '3p',   'be': 33.0,  'rsf': 0.444,  'role': 'secondary'},
            {'name': 'LMM',  'be': 1067,  'rsf': 0,      'role': 'auger'},
        ],
    },
    'Cr': {
        'lines': [
            {'name': '2p3/2','be': 574.0, 'rsf': 11.420, 'role': 'primary'},
            {'name': '2p1/2','be': 583.7, 'rsf': 5.760,  'role': 'secondary'},
            {'name': '3p',   'be': 43.5,  'rsf': 0.534,  'role': 'secondary'},
        ],
    },
    'Mn': {
        'lines': [
            {'name': '2p3/2','be': 641.0, 'rsf': 13.110, 'role': 'primary'},
            {'name': '2p1/2','be': 652.0, 'rsf': 6.620,  'role': 'secondary'},
        ],
    },
    'Fe': {
        'lines': [
            {'name': '2p3/2','be': 706.7, 'rsf': 16.420, 'role': 'primary'},
            {'name': '2p1/2','be': 720.3, 'rsf': 8.290,  'role': 'secondary'},
            {'name': '3p',   'be': 53.0,  'rsf': 0.730,  'role': 'secondary'},
            {'name': 'LMM',  'be': 781,   'rsf': 0,      'role': 'auger'},
        ],
    },
    'Co': {
        'lines': [
            {'name': '2p3/2','be': 778.2, 'rsf': 18.720, 'role': 'primary'},
            {'name': '2p1/2','be': 793.2, 'rsf': 9.450,  'role': 'secondary'},
        ],
    },
    'Ni': {
        'lines': [
            {'name': '2p3/2','be': 852.7, 'rsf': 21.180, 'role': 'primary'},
            {'name': '2p1/2','be': 870.0, 'rsf': 10.690, 'role': 'secondary'},
            {'name': '3p',   'be': 67.0,  'rsf': 0.888,  'role': 'secondary'},
        ],
    },
    'Cu': {
        'lines': [
            {'name': '2p3/2','be': 932.7, 'rsf': 23.780, 'role': 'primary'},
            {'name': '2p1/2','be': 952.5, 'rsf': 12.010, 'role': 'secondary'},
            {'name': '3s',   'be': 122.5, 'rsf': 0.385,  'role': 'secondary'},
            {'name': '3p',   'be': 75.5,  'rsf': 0.978,  'role': 'secondary'},
            {'name': 'LMM',  'be': 568,   'rsf': 0,      'role': 'auger'},
        ],
    },
    'Zn': {
        'lines': [
            {'name': '2p3/2','be': 1021.8,'rsf': 26.500, 'role': 'primary'},
            {'name': '2p1/2','be': 1044.9,'rsf': 13.380, 'role': 'secondary'},
            {'name': '3p',   'be': 88.6,  'rsf': 1.080,  'role': 'secondary'},
            {'name': 'LMM',  'be': 499,   'rsf': 0,      'role': 'auger'},
        ],
    },

    # ---------- 4d / 5d transition + post-transition ----------
    'Ga': {
        'lines': [
            {'name': '2p3/2','be': 1117.0,'rsf': 31.330, 'role': 'primary'},
            {'name': '3d',   'be': 18.7,  'rsf': 0.310,  'role': 'secondary'},
            {'name': '3p',   'be': 105.0, 'rsf': 1.310,  'role': 'secondary'},
        ],
    },
    'As': {
        'lines': [
            {'name': '3d',   'be': 41.7,  'rsf': 0.555,  'role': 'primary'},
            {'name': '3p',   'be': 141.0, 'rsf': 1.580,  'role': 'secondary'},
            {'name': '2p3/2','be': 1323.0,'rsf': 44.30,  'role': 'secondary'},
        ],
    },
    'Mo': {
        'lines': [
            {'name': '3d5/2','be': 228.0, 'rsf': 5.250,  'role': 'primary'},
            {'name': '3d3/2','be': 231.1, 'rsf': 3.510,  'role': 'secondary'},
            {'name': '3p',   'be': 395.0, 'rsf': 5.220,  'role': 'secondary'},
        ],
    },
    'Ag': {
        'lines': [
            {'name': '3d5/2','be': 368.3, 'rsf': 9.580,  'role': 'primary'},
            {'name': '3d3/2','be': 374.3, 'rsf': 6.440,  'role': 'secondary'},
            {'name': '3p',   'be': 573.0, 'rsf': 6.310,  'role': 'secondary'},
            {'name': 'MNN',  'be': 1130,  'rsf': 0,      'role': 'auger'},
        ],
    },
    'In': {
        'lines': [
            {'name': '3d5/2','be': 444.5, 'rsf': 12.310, 'role': 'primary'},
            {'name': '3d3/2','be': 452.0, 'rsf': 8.260,  'role': 'secondary'},
            {'name': '4d',   'be': 17.0,  'rsf': 0.430,  'role': 'secondary'},
        ],
    },
    'Sn': {
        'lines': [
            {'name': '3d5/2','be': 486.6, 'rsf': 13.510, 'role': 'primary'},
            {'name': '3d3/2','be': 494.9, 'rsf': 9.080,  'role': 'secondary'},
            {'name': '3p3/2','be': 715.0, 'rsf': 6.360,  'role': 'secondary'},
            {'name': '3p1/2','be': 757.0, 'rsf': 3.220,  'role': 'secondary'},
            {'name': '4d',   'be': 25.5,  'rsf': 0.534,  'role': 'secondary'},
            {'name': 'MNN',  'be': 1070,  'rsf': 0,      'role': 'auger'},
        ],
    },

    # ---------- Heavy elements ----------
    'Hf': {
        'lines': [
            {'name': '4f7/2','be': 14.5,  'rsf': 1.969,  'role': 'primary'},
            {'name': '4f5/2','be': 16.2,  'rsf': 1.520,  'role': 'secondary'},
            {'name': '4d',   'be': 213.5, 'rsf': 4.080,  'role': 'secondary'},
            {'name': '4p',   'be': 380,   'rsf': 4.640,  'role': 'secondary'},
        ],
    },
    'Ta': {
        'lines': [
            {'name': '4f7/2','be': 21.6,  'rsf': 2.220,  'role': 'primary'},
            {'name': '4f5/2','be': 23.5,  'rsf': 1.720,  'role': 'secondary'},
            {'name': '4d',   'be': 230.0, 'rsf': 4.380,  'role': 'secondary'},
        ],
    },
    'W': {
        'lines': [
            {'name': '4f7/2','be': 31.4,  'rsf': 2.480,  'role': 'primary'},
            {'name': '4f5/2','be': 33.6,  'rsf': 1.920,  'role': 'secondary'},
            {'name': '4d',   'be': 244.0, 'rsf': 4.740,  'role': 'secondary'},
        ],
    },
    'Pt': {
        'lines': [
            {'name': '4f7/2','be': 71.2,  'rsf': 3.560,  'role': 'primary'},
            {'name': '4f5/2','be': 74.5,  'rsf': 2.760,  'role': 'secondary'},
            {'name': '4d',   'be': 314.0, 'rsf': 5.840,  'role': 'secondary'},
        ],
    },
    'Au': {
        'lines': [
            {'name': '4f7/2','be': 84.0,  'rsf': 4.110,  'role': 'primary'},
            {'name': '4f5/2','be': 87.7,  'rsf': 3.170,  'role': 'secondary'},
            {'name': '4d',   'be': 335.0, 'rsf': 6.250,  'role': 'secondary'},
        ],
    },
}


# ===================================================================
# 자동 캘리브레이션 (C 1s 기준)
# ===================================================================
def auto_calibrate_c1s(detected_peaks, target_be=284.8, search_window=10.0):
    """
    C 1s를 찾아 target_be(284.8)으로 보정할 shift 값을 반환.

    Survey에서는 거의 항상 표면 탄소 오염(adventitious carbon)으로 인한
    C 1s가 285±5 eV 부근에 보임. 이를 기준으로 charging shift 추정.

    Returns:
        shift (float): 검출된 C1s에서 target까지의 보정값 (eV)
                        None이면 캘리브레이션 불가
    """
    candidates = [p for p in detected_peaks
                   if abs(p['be'] - target_be) <= search_window]
    if not candidates:
        return None
    # 가장 강한 후보 선택 (충전 시프트는 일관되므로)
    best = max(candidates, key=lambda p: p['intensity'])
    shift = target_be - best['be']
    return shift


# ===================================================================
# Survey 판별
# ===================================================================
def is_survey_scan(be_array, threshold_ev=500.0):
    """BE 범위가 threshold_ev 이상이면 Survey로 판별"""
    if len(be_array) == 0:
        return False
    return (be_array.max() - be_array.min()) >= threshold_ev


# ===================================================================
# 피크 검출 — Survey용 (단순하지만 보수적)
# ===================================================================
def detect_survey_peaks(be, counts, prominence_ratio=0.02, min_distance_ev=2.0):
    """
    Survey 스펙트럼에서 피크 검출.
    - prominence_ratio: 최대값의 비율 이상의 prominence
    - min_distance_ev: 피크 간 최소 거리 (eV)

    배경 보정은 단순화 (전체 평균 leading edge 차감 정도).
    """
    # 가벼운 스무딩
    win = max(7, len(counts) // 60)
    if win % 2 == 0: win += 1
    if win > len(counts):
        win = len(counts) - 1 if len(counts) % 2 else len(counts) - 2
    y_smooth = savgol_filter(counts, win, 3)

    # 단순 배경: 양 끝 일부의 평균을 직선으로 빼기
    edge_n = max(5, len(be) // 50)
    bg_left = np.mean(counts[:edge_n])
    bg_right = np.mean(counts[-edge_n:])
    # be가 오름차순이라고 가정 (작은 BE → 큰 BE)
    bg = np.linspace(bg_left, bg_right, len(be))
    y_corr = y_smooth - bg

    # step 추정 (eV per index)
    step_ev = abs(be[1] - be[0])
    distance = max(1, int(min_distance_ev / step_ev))

    prom = max(y_corr.max() * prominence_ratio, 1.0)
    peaks_idx, props = find_peaks(y_corr, prominence=prom, distance=distance)

    # 검출된 피크 정보
    detected = []
    for i in peaks_idx:
        detected.append({
            'be': float(be[i]),
            'intensity': float(y_corr[i]),
            'raw_intensity': float(counts[i]),
        })
    detected.sort(key=lambda p: -p['intensity'])  # 강도 큰 순
    return detected, bg


# ===================================================================
# 다중 피크 매칭으로 원소 식별
# ===================================================================
@dataclass
class ElementMatch:
    element: str
    matched_lines: list  # [{'name','be_expected','be_observed','intensity','role'}]
    score: float
    confidence: str  # 'high', 'medium', 'low'
    primary_line_be: float
    primary_line_intensity: float


def identify_elements(detected_peaks, be_range, tolerance_ev=4.0,
                       max_element_shift=5.0, secondary_tolerance=2.5):
    """
    검출된 피크들을 ELEMENT_DB와 매칭해 원소 식별 (자기일관성 검증).

    개선 사항 (v2):
    - 원소별로 "공통 시프트"를 추정해 모든 라인이 일관된 시프트로 매칭되는지 확인
    - tolerance_ev: primary 라인 매칭 허용 BE 오차 (관대하게)
    - secondary_tolerance: shift 적용 후 secondary 매칭 허용 오차
                            (DB 값 자체 오차 흡수)
    - max_element_shift: 원소 전체의 최대 허용 charging shift
    """
    be_min, be_max = be_range
    matches = []

    for elem, info in ELEMENT_DB.items():
        # BE 범위 안 라인만 후보로
        candidate_lines = [l for l in info['lines']
                            if be_min <= l['be'] <= be_max]
        if not candidate_lines:
            continue

        # primary 라인 찾기 (가장 강한 RSF + role=primary)
        primary_lines = [l for l in candidate_lines if l['role'] == 'primary']
        if not primary_lines:
            continue
        primary_line = primary_lines[0]

        # primary 매칭 후보 (tolerance 내)
        primary_candidates = [
            (pk, abs(pk['be'] - primary_line['be']))
            for pk in detected_peaks
            if abs(pk['be'] - primary_line['be']) <= tolerance_ev
        ]
        if not primary_candidates:
            continue

        # 각 primary 후보에 대해 시도
        best_attempt = None
        for primary_pk, _ in primary_candidates:
            # 이 매칭으로 추정되는 element shift
            element_shift = primary_pk['be'] - primary_line['be']

            # 이 시프트가 너무 크면 의심스러운 매칭
            if abs(element_shift) > max_element_shift:
                continue

            # 다른 라인들에 대해 같은 시프트로 매칭 시도
            matched = [{
                'name': primary_line['name'],
                'be_expected': primary_line['be'],
                'be_observed': primary_pk['be'],
                'be_shift': element_shift,
                'intensity': primary_pk['intensity'],
                'role': primary_line['role'],
                'rsf': primary_line.get('rsf', 0),
            }]

            for line in candidate_lines:
                if line['name'] == primary_line['name']:
                    continue
                expected_with_shift = line['be'] + element_shift
                # 이 시프트된 위치에 검출 피크가 있는지
                close = [p for p in detected_peaks
                          if abs(p['be'] - expected_with_shift) <= secondary_tolerance]
                if close:
                    closest = min(close, key=lambda p: abs(p['be'] - expected_with_shift))
                    matched.append({
                        'name': line['name'],
                        'be_expected': line['be'],
                        'be_observed': closest['be'],
                        'be_shift': closest['be'] - line['be'],
                        'intensity': closest['intensity'],
                        'role': line['role'],
                        'rsf': line.get('rsf', 0),
                    })

            # 이 시도의 점수 = 매칭된 라인 수
            attempt_score = len(matched)
            if (best_attempt is None or
                attempt_score > best_attempt['score']):
                best_attempt = {
                    'matched': matched,
                    'shift': element_shift,
                    'score': attempt_score,
                }

        if best_attempt is None:
            continue

        matched = best_attempt['matched']
        n_secondary = sum(1 for m in matched if m['role'] in ('secondary', 'auger'))

        if n_secondary >= 2:
            confidence = 'high'
            score = 1.0 + n_secondary * 0.2
        elif n_secondary == 1:
            confidence = 'medium'
            score = 0.7
        else:
            confidence = 'low'
            score = 0.4

        matches.append(ElementMatch(
            element=elem,
            matched_lines=matched,
            score=score,
            confidence=confidence,
            primary_line_be=matched[0]['be_observed'],
            primary_line_intensity=matched[0]['intensity'],
        ))

    matches.sort(key=lambda m: -m.primary_line_intensity)
    return matches


# ===================================================================
# Atomic % 정량 (Phase B)
# ===================================================================
def quantify_atomic_percent(matches, only_high_confidence=True):
    """
    각 원소의 atomic % 계산.
    공식: at% = (I / RSF) / Σ (I_j / RSF_j) × 100

    - I: primary peak의 (배경 제거된) intensity (간이값으로 peak height 사용)
    - RSF: 해당 라인의 sensitivity factor
    - low confidence는 기본적으로 제외 (불확실한 것 포함하면 정량 망가짐)
    """
    candidates = [m for m in matches
                   if not only_high_confidence or m.confidence != 'low']

    if not candidates:
        return []

    # 각 원소의 normalized intensity (I / RSF)
    contributions = []
    for m in candidates:
        primary = m.matched_lines[0]
        rsf = primary.get('rsf', 0)
        if rsf <= 0:
            # RSF 없으면 정량 불가 → 표시는 하되 % 계산 안 함
            contributions.append((m, None))
            continue
        norm_I = primary['intensity'] / rsf
        contributions.append((m, norm_I))

    total = sum(v for _, v in contributions if v is not None)
    if total <= 0:
        return [(m, None) for m, _ in contributions]

    results = []
    for m, norm_I in contributions:
        if norm_I is None:
            results.append((m, None))
        else:
            atomic_pct = 100 * norm_I / total
            results.append((m, atomic_pct))
    return results


# ===================================================================
# 메인 파이프라인
# ===================================================================
def analyze_survey(be, counts, tolerance_ev=4.0, auto_calibrate=False):
    """
    Survey 자동 분석 메인 함수.

    Parameters:
        be, counts: 데이터
        tolerance_ev: primary 라인 매칭 허용 BE 오차 (4 eV 기본)
                       — 충전(charging) 시프트 흡수 위해 관대하게 설정
        auto_calibrate: True면 C 1s 자동 캘리브레이션 시도
                         (False가 기본 — 자기일관성 매칭이 시프트 자동 처리)

    내부 알고리즘:
    - 각 원소에 대해 "원소 자기일관성" 검증
    - primary 라인 매칭 → 그 시프트로 secondary들도 매칭되는지 확인
    - 일관된 시프트로 여러 라인 매칭되면 high confidence
    """
    # 1) 피크 검출
    detected_peaks, bg = detect_survey_peaks(be, counts)

    if len(detected_peaks) == 0:
        return {
            'success': False,
            'reason': 'No peaks detected in survey',
            'be': be, 'counts': counts, 'background': bg,
        }

    # 2) 자동 캘리브레이션 (C 1s 기준)
    calibration_info = {'applied': False, 'shift': 0.0,
                          'method': 'none', 'note': ''}
    be_cal = be
    detected_cal = detected_peaks
    if auto_calibrate:
        shift = auto_calibrate_c1s(detected_peaks)
        if shift is not None:
            be_cal = be + shift
            # 검출 피크 위치도 모두 shift
            detected_cal = [
                {**p, 'be': p['be'] + shift} for p in detected_peaks
            ]
            calibration_info = {
                'applied': True, 'shift': shift,
                'method': 'C1s auto (target=284.8 eV)',
                'note': (f"Detected C 1s at {284.8 - shift:.2f} eV, "
                         f"shifted by {shift:+.2f} eV"),
            }

    # 3) 원소 식별 (캘리브레이션된 좌표로)
    matches = identify_elements(detected_cal,
                                  (be_cal.min(), be_cal.max()),
                                  tolerance_ev=tolerance_ev)

    # 4) 정량 (high + medium만 사용)
    quantification = quantify_atomic_percent(matches,
                                                only_high_confidence=True)

    return {
        'success': True,
        'mode': 'survey',
        'be': be_cal, 'counts': counts, 'background': bg,
        'be_original': be,  # 원본도 보존
        'detected_peaks': detected_cal,
        'matches': matches,
        'quantification': quantification,
        'n_elements': len([m for m in matches if m.confidence != 'low']),
        'calibration_info': calibration_info,
    }
