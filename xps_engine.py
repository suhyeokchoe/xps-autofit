"""
XPS Auto-fitting Engine v0.2
================================================
- 2차 미분 기반 shoulder 감지
- 다중 모델 AIC 기반 피크 개수 자동 결정 (overfitting 방지)
- 원소별 도메인 prior
- Calibration: C1s 기준 shift 옵션
"""
import csv
import io
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from pathlib import Path

# -------------------------------------------------------------------
# 원소별 도메인 prior
# -------------------------------------------------------------------
ELEMENT_PRIORS = {
    'F1s':   (680, 695, 0.8, 3.5),
    'C1s':   (280, 295, 0.7, 2.5),
    'O1s':   (525, 540, 0.8, 3.0),
    'Sn3d':  (480, 500, 0.7, 2.0),
    'Cu2p':  (925, 970, 0.8, 3.0),
    'N1s':   (395, 410, 0.8, 2.5),
    'Si2p':  (95, 110, 0.6, 2.5),
}


def detect_region(be_min, be_max):
    for name, (lo, hi, _, _) in ELEMENT_PRIORS.items():
        if be_min >= lo - 5 and be_max <= hi + 5:
            return name
    return None


# -------------------------------------------------------------------
# 데이터 로딩
# -------------------------------------------------------------------
def load_xps_csv(path_or_text, source_name='uploaded'):
    """CSV 파일 경로 or CSV 텍스트 모두 지원"""
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

    if meta['region'] is None:
        meta['region'] = detect_region(be.min(), be.max()) or 'unknown'
    return be, counts, meta


# -------------------------------------------------------------------
# Shirley background
# -------------------------------------------------------------------
def shirley_background(x, y, max_iter=60, tol=1e-6):
    B = np.zeros_like(y, dtype=float)
    I_L, I_R = y[0], y[-1]
    for _ in range(max_iter):
        cum = np.zeros_like(y, dtype=float)
        for i in range(1, len(y)):
            cum[i] = cum[i-1] + 0.5 * (
                (y[i-1] - I_R - B[i-1]) + (y[i] - I_R - B[i])
            ) * (x[i] - x[i-1])
        denom = cum[-1] if abs(cum[-1]) > 1e-12 else 1.0
        k = (I_L - I_R) / denom
        B_new = I_R + k * cum
        if np.max(np.abs(B_new - B)) < tol * max(abs(I_L - I_R), 1.0):
            return B_new
        B = B_new
    return B


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


# -------------------------------------------------------------------
# 피크 감지: 1차(주 피크) + 2차 미분(shoulder)
# -------------------------------------------------------------------
def detect_peaks_v2(x, y_corr, region=None):
    win = max(7, len(y_corr) // 25)
    if win % 2 == 0: win += 1

    y_smooth = savgol_filter(y_corr, win, 3)
    d2 = savgol_filter(y_corr, win, 3, deriv=2)

    # 1차: 명확한 봉우리
    prom = max(y_smooth) * 0.05
    main_idx, _ = find_peaks(y_smooth, prominence=prom,
                              distance=max(5, len(x)//40))
    # 2차: shoulder (보수적 threshold)
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
# 단일 n_peaks로 피팅
# -------------------------------------------------------------------
def fit_n_peaks(x, y_corr, n, init_centers, region=None):
    if region in ELEMENT_PRIORS:
        _, _, fwhm_min, fwhm_max = ELEMENT_PRIORS[region]
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
        popt, pcov = curve_fit(multi_pv, x, y_corr,
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
                'aic': aic, 'n_peaks': n}
    except Exception:
        return None


# -------------------------------------------------------------------
# 자동 파이프라인 with AIC 모델 선택
# -------------------------------------------------------------------
def auto_fit_v2(be, counts, meta=None, max_peaks=4):
    meta = meta or {}
    region = meta.get('region')

    bg = shirley_background(be, counts)
    y_corr = counts - bg

    peaks_idx, y_smooth = detect_peaks_v2(be, y_corr, region)
    if len(peaks_idx) == 0:
        return {'success': False, 'reason': 'No peaks detected',
                'be': be, 'counts': counts, 'background': bg}

    init_centers = [float(be[i]) for i in peaks_idx]

    max_try = min(max_peaks, len(init_centers))
    max_try = max(max_try, 1)

    trials = []
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

    best = min(trials, key=lambda r: r['aic'])
    if best['r2'] < 0.9:
        return {'success': False, 'reason': f'Best R²={best["r2"]:.3f} too low',
                'be': be, 'counts': counts, 'background': bg}

    components = []
    for i in range(best['n_peaks']):
        a, c, f, e = best['popt'][i*4:i*4+4]
        comp_y = pseudo_voigt(be, a, c, f, e)
        components.append({
            'amplitude': float(a), 'position': float(c),
            'fwhm': float(f), 'eta': float(e),
            'area': float(abs(np.trapezoid(comp_y, be))),
            'curve': comp_y,
        })
    components.sort(key=lambda c: c['position'])
    total_area = sum(c['area'] for c in components) or 1
    for c in components:
        c['area_pct'] = 100 * c['area'] / total_area

    trial_summary = [{'n_peaks': t['n_peaks'], 'r2': t['r2'],
                      'rms': t['rms'], 'aic': t['aic']} for t in trials]

    return {
        'success': True, 'meta': meta,
        'region': region,
        'be': be, 'counts': counts, 'background': bg,
        'y_corrected': y_corr, 'y_fit': best['y_fit'],
        'components': components,
        'r_squared': best['r2'], 'rms': best['rms'], 'aic': best['aic'],
        'n_peaks': best['n_peaks'],
        'trials': trial_summary,
    }


# -------------------------------------------------------------------
# Calibration
# -------------------------------------------------------------------
def calibrate_shift(be, shift):
    return be + shift
