"""
XPS AutoFit — Streamlit Web App
================================================
MVP 기능:
- CSV/TXT 업로드 (CasaXPS export 형식 + 단순 2열 형식 지원)
- 자동 피팅 + AIC 기반 피크 수 자동 결정
- 사용자가 피크 수를 override 가능
- 결과 시각화 + 파라미터 테이블
- CSV/PNG 다운로드

실행: streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from xps_engine import (
    load_xps_csv, auto_fit_v2, fit_n_peaks, shirley_background,
    detect_peaks_v2, pseudo_voigt, calibrate_shift,
    ELEMENT_PRIORS, detect_region,
)

# -------------------------------------------------------------------
# 페이지 설정
# -------------------------------------------------------------------
st.set_page_config(
    page_title="XPS AutoFit",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stMetric { background: #f0f2f6; padding: 0.8rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# 헤더
# -------------------------------------------------------------------
st.title("📊 XPS AutoFit")
st.caption("자동 XPS 피팅 도구 · Shirley background + pseudo-Voigt + AIC 모델 선택")


# -------------------------------------------------------------------
# 사이드바: 업로드 + 옵션
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📤 데이터 업로드")
    uploaded = st.file_uploader(
        "CSV 또는 TXT 파일",
        type=['csv', 'txt'],
        help="CasaXPS export 형식 또는 2열(BE, Counts) 단순 CSV"
    )

    st.caption(
        "💡 **.xls 사용자**: Excel에서 `다른 이름으로 저장` → CSV(.csv)로 변환 후 업로드하세요."
    )

    st.divider()
    st.header("⚙️ 피팅 옵션")
    max_peaks = st.slider("최대 탐색 피크 수", 1, 8, 4,
                           help="자동 모델 선택 시 고려할 최대 피크 개수")

    manual_n = st.number_input(
        "피크 수 수동 지정 (0=자동)",
        min_value=0, max_value=8, value=0,
        help="0이면 AIC로 자동 선택, 1 이상이면 해당 개수로 강제"
    )

    st.divider()
    st.header("🎯 Calibration")
    apply_cal = st.checkbox("BE offset 적용", value=False)
    cal_shift = st.number_input(
        "Shift (eV)", value=0.0, step=0.05, format="%.2f",
        disabled=not apply_cal,
        help="C1s = 284.8 eV 기준 보정 등"
    )

    st.divider()
    with st.expander("ℹ️ 지원 데이터 포맷"):
        st.markdown("""
**CasaXPS 형식** (이 앱의 기본):
```
Binding Energy (E),,,,Backgnd.
eV,,Counts / s,,Counts / s
698.08,,28442.2,,0
697.98,,28008.9,,0
...
```

**간단한 2열 형식**도 지원:
```
BE,Counts
698.08,28442.2
...
```
""")


# -------------------------------------------------------------------
# 메인: 결과 영역
# -------------------------------------------------------------------
if uploaded is None:
    st.info("👈 사이드바에서 XPS 데이터 파일을 업로드하세요.")

    with st.expander("🔬 이 앱이 하는 일", expanded=True):
        st.markdown("""
1. **배경 제거** — iterative Shirley background
2. **피크 자동 감지** — 1차 미분(주 피크) + 2차 미분(shoulder)
3. **최적 모델 선택** — AIC(Akaike Information Criterion)로 피크 수 결정
4. **라인쉐입 피팅** — pseudo-Voigt (= CasaXPS의 SGL)
5. **결과 export** — 파라미터 테이블 CSV, 시각화 PNG

⚠️ **중요**: R²가 높다고 올바른 모델이 아닙니다.
결과는 항상 도메인 지식으로 검증하세요 (피크 위치, FWHM, 면적비가 화학적으로 합당한가?).
        """)
    st.stop()


# 데이터 로딩
try:
    raw_bytes = uploaded.read()
    text = raw_bytes.decode('utf-8-sig', errors='replace')
    be, counts, meta = load_xps_csv(text, source_name=uploaded.name)
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

# Calibration
if apply_cal and cal_shift != 0:
    be = calibrate_shift(be, cal_shift)
    meta['calibrated'] = f"shift={cal_shift:+.2f} eV"


# -------------------------------------------------------------------
# 상단 메타 정보
# -------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("파일", meta['source_file'])
col2.metric("Region", meta.get('region', 'auto'))
col3.metric("데이터 포인트", len(be))
col4.metric("BE 범위",
             f"{be.min():.1f} – {be.max():.1f} eV")


# -------------------------------------------------------------------
# 피팅 실행
# -------------------------------------------------------------------
with st.spinner("피팅 중..."):
    if manual_n == 0:
        result = auto_fit_v2(be, counts, meta, max_peaks=max_peaks)
    else:
        # 수동 n: 초기값 감지 후 n개 강제
        bg = shirley_background(be, counts)
        y_corr = counts - bg
        peaks_idx, _ = detect_peaks_v2(be, y_corr, meta.get('region'))
        if len(peaks_idx) == 0:
            st.error("피크를 감지할 수 없습니다.")
            st.stop()
        init_centers = sorted([float(be[i]) for i in peaks_idx[:manual_n]])
        # 부족하면 max값 부근에 추가
        while len(init_centers) < manual_n:
            init_centers.append(float(be[int(np.argmax(y_corr))]) +
                                0.5 * (len(init_centers) - manual_n // 2))
        fit = fit_n_peaks(be, y_corr, manual_n,
                          sorted(init_centers), meta.get('region'))
        if fit is None:
            st.error(f"{manual_n}개 피크로 피팅 실패.")
            st.stop()
        # 결과 포맷 맞추기
        components = []
        for i in range(manual_n):
            a, c, f, e = fit['popt'][i*4:i*4+4]
            comp_y = pseudo_voigt(be, a, c, f, e)
            components.append({
                'amplitude': float(a), 'position': float(c),
                'fwhm': float(f), 'eta': float(e),
                'area': float(abs(np.trapezoid(comp_y, be))),
                'curve': comp_y,
            })
        components.sort(key=lambda c: c['position'])
        total = sum(c['area'] for c in components) or 1
        for c in components:
            c['area_pct'] = 100 * c['area'] / total
        result = {
            'success': True, 'meta': meta, 'region': meta.get('region'),
            'be': be, 'counts': counts, 'background': bg,
            'y_corrected': y_corr, 'y_fit': fit['y_fit'],
            'components': components,
            'r_squared': fit['r2'], 'rms': fit['rms'], 'aic': fit['aic'],
            'n_peaks': manual_n,
            'trials': None,
        }


if not result['success']:
    st.error(f"피팅 실패: {result['reason']}")
    # 그래도 raw data는 보여주기
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(be, counts, 'o', mfc='none', mec='black', ms=3)
    ax.set_xlabel('Binding Energy (eV)'); ax.set_ylabel('Counts / s')
    ax.invert_xaxis()
    st.pyplot(fig)
    st.stop()


# -------------------------------------------------------------------
# 결과: 메트릭
# -------------------------------------------------------------------
st.subheader("📈 피팅 결과")
m1, m2, m3, m4 = st.columns(4)
m1.metric("모델", f"{result['n_peaks']}개 피크")
m2.metric("R²", f"{result['r_squared']:.4f}")
m3.metric("RMS", f"{result['rms']:.1f}")
m4.metric("AIC", f"{result['aic']:.1f}")

if result['r_squared'] < 0.97:
    st.warning("⚠️ R² < 0.97: 피팅 품질이 낮습니다. 피크 수를 수동으로 조정해보세요.")


# -------------------------------------------------------------------
# 메인 플롯
# -------------------------------------------------------------------
colors = ['#e63946', '#457b9d', '#06a77d', '#f4a261',
          '#9b5de5', '#f15bb5', '#00bbf9']

fig, (ax_main, ax_resid) = plt.subplots(
    2, 1, figsize=(12, 7),
    gridspec_kw={'height_ratios': [4, 1]}, sharex=True
)

# 상단: 피팅
ax_main.plot(be, counts, 'o', mfc='none', mec='black',
              ms=3, label='Experimental', zorder=3)
ax_main.plot(be, result['background'], '--', color='gray',
              lw=1, label='Shirley BG', zorder=2)
total_with_bg = result['background'] + result['y_fit']
ax_main.plot(be, total_with_bg, '-', color='red', lw=1.5,
              label='Envelope', zorder=4)

for i, comp in enumerate(result['components']):
    color = colors[i % len(colors)]
    comp_full = result['background'] + comp['curve']
    ax_main.fill_between(be, result['background'], comp_full,
                          alpha=0.3, color=color)
    ax_main.plot(be, comp_full, '-', color=color, lw=1.2,
                  label=f"Peak {i+1}: {comp['position']:.2f} eV "
                        f"({comp['area_pct']:.1f}%)")

ax_main.set_ylabel('Counts / s')
ax_main.set_title(f"{meta['source_file']} — {meta.get('region', '?')}")
ax_main.invert_xaxis()
ax_main.legend(loc='upper left', fontsize=9)
ax_main.grid(alpha=0.2)

# 하단: residual
resid = result['counts'] - (result['background'] + result['y_fit'])
ax_resid.plot(be, resid, 'o-', mfc='none', mec='gray',
               color='gray', ms=2, lw=0.5)
ax_resid.axhline(0, color='red', lw=0.8)
ax_resid.set_xlabel('Binding Energy (eV)')
ax_resid.set_ylabel('Residual')
ax_resid.invert_xaxis()
ax_resid.grid(alpha=0.2)

plt.tight_layout()
st.pyplot(fig)


# -------------------------------------------------------------------
# 파라미터 테이블
# -------------------------------------------------------------------
st.subheader("🔢 컴포넌트 파라미터")
df = pd.DataFrame([
    {
        'Peak': f"Peak {i+1}",
        'Position (eV)': round(c['position'], 3),
        'FWHM (eV)': round(c['fwhm'], 3),
        'Amplitude': round(c['amplitude'], 1),
        'GL ratio (η)': round(c['eta'], 3),
        'Area': round(c['area'], 1),
        'Area (%)': round(c['area_pct'], 2),
    }
    for i, c in enumerate(result['components'])
])
st.dataframe(df, use_container_width=True)


# -------------------------------------------------------------------
# 모델 선택 이력 (자동 모드일 때)
# -------------------------------------------------------------------
if result.get('trials'):
    with st.expander("🔎 AIC 기반 모델 선택 이력"):
        trial_df = pd.DataFrame(result['trials'])
        trial_df['선택됨'] = trial_df['n_peaks'] == result['n_peaks']
        trial_df['선택됨'] = trial_df['선택됨'].apply(lambda x: '★' if x else '')
        st.dataframe(trial_df, use_container_width=True)
        st.caption(
            "AIC = N·ln(SSR/N) + 2k. **낮을수록 좋음.** "
            "R²는 피크 수가 늘어날수록 항상 증가하지만 AIC는 "
            "파라미터 수에 페널티를 부여해 과적합을 방지합니다."
        )


# -------------------------------------------------------------------
# 다운로드
# -------------------------------------------------------------------
st.divider()
st.subheader("💾 다운로드")

dl1, dl2, dl3 = st.columns(3)

# 1) 파라미터 CSV
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
dl1.download_button(
    "📋 파라미터 CSV",
    data=csv_buf.getvalue(),
    file_name=f"{meta['source_file']}_fit_params.csv",
    mime='text/csv',
    use_container_width=True,
)

# 2) 피팅 곡선 CSV (재현성용)
curves_df = pd.DataFrame({'BE_eV': be, 'Counts_exp': counts,
                          'Background': result['background'],
                          'Envelope': result['background'] + result['y_fit']})
for i, c in enumerate(result['components']):
    curves_df[f'Peak_{i+1}'] = result['background'] + c['curve']
curves_buf = io.StringIO()
curves_df.to_csv(curves_buf, index=False)
dl2.download_button(
    "📊 피팅 곡선 CSV",
    data=curves_buf.getvalue(),
    file_name=f"{meta['source_file']}_fit_curves.csv",
    mime='text/csv',
    use_container_width=True,
)

# 3) PNG
png_buf = io.BytesIO()
fig.savefig(png_buf, format='png', dpi=180, bbox_inches='tight')
dl3.download_button(
    "🖼️ 플롯 PNG",
    data=png_buf.getvalue(),
    file_name=f"{meta['source_file']}_fit.png",
    mime='image/png',
    use_container_width=True,
)


# -------------------------------------------------------------------
# 푸터
# -------------------------------------------------------------------
st.divider()
st.caption(
    "XPS AutoFit · Phase 0 prototype · "
    "*Note: 자동 피팅은 도메인 지식 검증과 함께 사용하세요.*"
)
