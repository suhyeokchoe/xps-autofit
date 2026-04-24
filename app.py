"""
XPS AutoFit — Streamlit Web App (v0.3)
================================================
v0.2 대비:
- 스핀-오빗 doublet 자동 처리 (Sn 3d, Cu 2p 등)
- BE축 관례(큰→작은) 모든 출력에 강제
- Region 수동 선택 옵션
- 다운로드 CSV도 BE 내림차순

실행: streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from xps_engine import (
    load_xps_csv, auto_fit_v3, fit_n_peaks, fit_n_doublets,
    shirley_background, detect_peaks_v2, pseudo_voigt,
    calibrate_shift,
    ELEMENT_PRIORS, DOUBLET_PRIORS, is_doublet,
)

st.set_page_config(page_title="XPS AutoFit", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stMetric { background: #f0f2f6; padding: 0.8rem; border-radius: 0.5rem; }
    .doublet-badge { background: #06a77d; color: white;
                     padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }
    .singlet-badge { background: #457b9d; color: white;
                     padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

st.title("📊 XPS AutoFit")
st.caption("자동 XPS 피팅 · v0.3 · Spin-orbit doublet 지원")


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📤 데이터 업로드")
    uploaded = st.file_uploader(
        "CSV 또는 TXT 파일",
        type=['csv', 'txt'],
        help="CasaXPS export 형식 또는 2열(BE, Counts) CSV"
    )
    st.caption("💡 .xls는 Excel에서 CSV로 저장 후 업로드")

    st.divider()
    st.header("🧪 Region 설정")
    all_regions = ['auto'] + list(ELEMENT_PRIORS.keys()) + list(DOUBLET_PRIORS.keys())
    region_override = st.selectbox(
        "Region (원소)", all_regions, index=0,
        help="auto면 BE 범위로 자동 판별"
    )

    st.divider()
    st.header("⚙️ 피팅 옵션")
    max_peaks = st.slider("최대 탐색 (피크/상태 수)", 1, 6, 3)
    manual_n = st.number_input(
        "수동 지정 (0=자동)",
        min_value=0, max_value=6, value=0
    )
    force_singlet = st.checkbox(
        "Doublet 강제 해제 (singlet만)",
        value=False
    )

    st.divider()
    st.header("🎯 Calibration")
    apply_cal = st.checkbox("BE offset 적용", value=False)
    cal_shift = st.number_input(
        "Shift (eV)", value=0.0, step=0.05, format="%.2f",
        disabled=not apply_cal
    )

    st.divider()
    with st.expander("ℹ️ 지원 원소"):
        st.markdown("**Singlet**: F1s, C1s, O1s, N1s")
        st.markdown("**Doublet 2p (2:1)**: Cu, Ti, Si, Fe, Ni")
        st.markdown("**Doublet 3d (3:2)**: Sn, In, Mo, Ag")
        st.markdown("**Doublet 4f (4:3)**: Au, W")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if uploaded is None:
    st.info("👈 사이드바에서 XPS 데이터 파일을 업로드하세요.")
    with st.expander("🆕 v0.3 업데이트 (2주차)", expanded=True):
        st.markdown("""
- ✅ **스핀-오빗 doublet 자동 처리**: Sn 3d, Cu 2p 등
- ✅ **물리 제약 강제**: ΔBE · 면적비 · 공유 FWHM/η
- ✅ **BE축 관례 준수**: 모든 플롯·CSV에서 큰→작은
- ✅ **Region 수동 선택**: 자동 판별 실패 시 override
        """)
    st.stop()


# Load
try:
    raw_bytes = uploaded.read()
    text = raw_bytes.decode('utf-8-sig', errors='replace')
    be, counts, meta = load_xps_csv(text, source_name=uploaded.name)
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

if region_override != 'auto':
    meta['region'] = region_override

if apply_cal and cal_shift != 0:
    be = calibrate_shift(be, cal_shift)
    meta['calibrated'] = f"shift={cal_shift:+.2f} eV"


# Meta display
region = meta.get('region', 'unknown')
is_dbl_region = is_doublet(region) and not force_singlet
mode_badge = ('<span class="doublet-badge">DOUBLET</span>' if is_dbl_region
              else '<span class="singlet-badge">SINGLET</span>')

col1, col2, col3, col4 = st.columns(4)
col1.metric("파일", meta['source_file'])
col2.markdown(f"**Region**: {region}  \n{mode_badge}", unsafe_allow_html=True)
col3.metric("Points", len(be))
col4.metric("BE 범위", f"{be.max():.1f} → {be.min():.1f} eV")


# Fit
with st.spinner("피팅 중..."):
    eff_meta = dict(meta)
    if force_singlet and region in DOUBLET_PRIORS:
        eff_meta['region'] = 'unknown'

    if manual_n == 0:
        result = auto_fit_v3(be, counts, eff_meta, max_peaks=max_peaks)
    else:
        bg = shirley_background(be, counts)
        y_corr = counts - bg
        peaks_idx, _ = detect_peaks_v2(be, y_corr, eff_meta.get('region'))
        if len(peaks_idx) == 0:
            st.error("피크 감지 실패"); st.stop()
        init_centers = sorted([float(be[i]) for i in peaks_idx])

        if is_doublet(eff_meta.get('region')) and not force_singlet:
            ranked = sorted(init_centers,
                            key=lambda c: -y_corr[int(np.argmin(np.abs(be - c)))])
            centers = sorted(ranked[:manual_n])
            while len(centers) < manual_n:
                centers.append(centers[-1] - 2.0 if centers else float(be[np.argmax(y_corr)]))
            fit = fit_n_doublets(be, y_corr, manual_n, sorted(centers),
                                  eff_meta['region'])
            if fit is None:
                st.error(f"{manual_n}개 상태 doublet 피팅 실패"); st.stop()
            _, _, _, _, dBE, ar = DOUBLET_PRIORS[eff_meta['region']]
            components = []
            for i in range(manual_n):
                amp_m, c_m, fwhm, eta = fit['popt'][i*4:i*4+4]
                cm = pseudo_voigt(be, amp_m, c_m, fwhm, eta)
                components.append({
                    'amplitude': float(amp_m), 'position': float(c_m),
                    'fwhm': float(fwhm), 'eta': float(eta),
                    'area': float(abs(np.trapezoid(cm, be))),
                    'curve': cm, 'label': f'State {i+1} (main)',
                })
                amp_nv = amp_m / ar; c_nv = c_m + dBE
                cn = pseudo_voigt(be, amp_nv, c_nv, fwhm, eta)
                components.append({
                    'amplitude': float(amp_nv), 'position': float(c_nv),
                    'fwhm': float(fwhm), 'eta': float(eta),
                    'area': float(abs(np.trapezoid(cn, be))),
                    'curve': cn, 'label': f'State {i+1} (minor)',
                })
            result_mode = 'doublet'
        else:
            ranked = sorted(init_centers,
                            key=lambda c: -y_corr[int(np.argmin(np.abs(be - c)))])
            centers = sorted(ranked[:manual_n])
            while len(centers) < manual_n:
                centers.append(float(be[int(np.argmax(y_corr))]) +
                               0.5 * (len(centers) - manual_n // 2))
            fit = fit_n_peaks(be, y_corr, manual_n, sorted(centers),
                               eff_meta.get('region'))
            if fit is None:
                st.error(f"{manual_n}개 피크 피팅 실패"); st.stop()
            components = []
            for i in range(manual_n):
                a, c, f, e = fit['popt'][i*4:i*4+4]
                cm = pseudo_voigt(be, a, c, f, e)
                components.append({
                    'amplitude': float(a), 'position': float(c),
                    'fwhm': float(f), 'eta': float(e),
                    'area': float(abs(np.trapezoid(cm, be))),
                    'curve': cm, 'label': f'Peak {i+1}',
                })
            result_mode = 'singlet'

        components.sort(key=lambda c: -c['position'])
        for i, c in enumerate(components):
            if c['label'].startswith('Peak '):
                c['label'] = f'Peak {i+1}'
        total = sum(c['area'] for c in components) or 1
        for c in components:
            c['area_pct'] = 100 * c['area'] / total

        result = {
            'success': True, 'meta': meta, 'region': eff_meta.get('region'),
            'mode': result_mode, 'be': be, 'counts': counts,
            'background': bg, 'y_corrected': y_corr, 'y_fit': fit['y_fit'],
            'components': components,
            'r_squared': fit['r2'], 'rms': fit['rms'], 'aic': fit['aic'],
            'n_peaks': manual_n, 'trials': None,
            'doublet_info': {'delta_BE': fit.get('delta_BE'),
                              'area_ratio': fit.get('area_ratio')}
                              if result_mode == 'doublet' else None,
        }

if not result['success']:
    st.error(f"피팅 실패: {result['reason']}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(be, counts, 'o', mfc='none', mec='black', ms=3)
    ax.set_xlabel('Binding Energy (eV)'); ax.set_ylabel('Counts / s')
    ax.invert_xaxis()
    st.pyplot(fig); st.stop()


# Metrics
st.subheader("📈 피팅 결과")
mode_info = f"{result['n_peaks']}개 {'화학상태 (doublet)' if result['mode'] == 'doublet' else '피크 (singlet)'}"
m1, m2, m3, m4 = st.columns(4)
m1.metric("모델", mode_info)
m2.metric("R²", f"{result['r_squared']:.4f}")
m3.metric("RMS", f"{result['rms']:.1f}")
m4.metric("AIC", f"{result['aic']:.1f}")

if result.get('doublet_info'):
    di = result['doublet_info']
    st.info(f"🔗 **Doublet 물리 제약 적용**: ΔBE = {di['delta_BE']} eV, "
             f"면적비 main:minor = {di['area_ratio']}:1")

if result['r_squared'] < 0.97:
    st.warning("⚠️ R² < 0.97: 피크/상태 수 또는 region 설정 재확인 권장")


# Plot
colors = ['#e63946', '#457b9d', '#06a77d', '#f4a261',
          '#9b5de5', '#f15bb5', '#00bbf9']

fig, (ax_main, ax_resid) = plt.subplots(
    2, 1, figsize=(12, 7),
    gridspec_kw={'height_ratios': [4, 1]}, sharex=True
)

ax_main.plot(be, counts, 'o', mfc='none', mec='black', ms=3,
              label='Experimental', zorder=3)
ax_main.plot(be, result['background'], '--', color='gray', lw=1,
              label='Shirley BG', zorder=2)
total_with_bg = result['background'] + result['y_fit']
ax_main.plot(be, total_with_bg, '-', color='red', lw=1.5,
              label='Envelope', zorder=4)

if result['mode'] == 'doublet':
    state_colors = {}
    for comp in result['components']:
        state_key = ' '.join(comp['label'].split(' ')[:2])
        if state_key not in state_colors:
            state_colors[state_key] = colors[len(state_colors) % len(colors)]
        color = state_colors[state_key]
        is_main = '(main)' in comp['label']
        comp_full = result['background'] + comp['curve']
        alpha = 0.4 if is_main else 0.2
        ls = '-' if is_main else '--'
        lw = 1.3 if is_main else 0.9
        ax_main.fill_between(be, result['background'], comp_full,
                              alpha=alpha, color=color)
        ax_main.plot(be, comp_full, ls=ls, color=color, lw=lw,
                      label=f"{comp['label']}: {comp['position']:.2f} eV "
                            f"({comp['area_pct']:.1f}%)")
else:
    for i, comp in enumerate(result['components']):
        color = colors[i % len(colors)]
        comp_full = result['background'] + comp['curve']
        ax_main.fill_between(be, result['background'], comp_full,
                              alpha=0.3, color=color)
        ax_main.plot(be, comp_full, '-', color=color, lw=1.2,
                      label=f"{comp['label']}: {comp['position']:.2f} eV "
                            f"({comp['area_pct']:.1f}%)")

ax_main.set_ylabel('Counts / s')
ax_main.set_title(f"{meta['source_file']} — {region} ({result['mode']})")
ax_main.invert_xaxis()   # XPS 관례
ax_main.legend(loc='upper right', fontsize=9)
ax_main.grid(alpha=0.2)

resid = result['counts'] - (result['background'] + result['y_fit'])
ax_resid.plot(be, resid, 'o-', mfc='none', mec='gray', color='gray',
               ms=2, lw=0.5)
ax_resid.axhline(0, color='red', lw=0.8)
ax_resid.set_xlabel('Binding Energy (eV)')
ax_resid.set_ylabel('Residual')
ax_resid.invert_xaxis()  # XPS 관례
ax_resid.grid(alpha=0.2)

plt.tight_layout()
st.pyplot(fig)


# Table
st.subheader("🔢 컴포넌트 파라미터")
df = pd.DataFrame([
    {
        'Component': c['label'],
        'Position (eV)': round(c['position'], 3),
        'FWHM (eV)': round(c['fwhm'], 3),
        'Amplitude': round(c['amplitude'], 1),
        'GL ratio (η)': round(c['eta'], 3),
        'Area': round(c['area'], 1),
        'Area (%)': round(c['area_pct'], 2),
    }
    for c in result['components']
])
st.dataframe(df, use_container_width=True, hide_index=True)


if result.get('trials'):
    with st.expander("🔎 AIC 기반 모델 선택 이력"):
        trial_df = pd.DataFrame(result['trials'])
        trial_df['선택됨'] = trial_df.apply(
            lambda row: '★' if (row['n_peaks'] == result['n_peaks']
                                and row['mode'] == result['mode']) else '',
            axis=1
        )
        st.dataframe(trial_df, use_container_width=True, hide_index=True)


# Download (BE 내림차순)
st.divider()
st.subheader("💾 다운로드")
dl1, dl2, dl3 = st.columns(3)

csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
dl1.download_button("📋 파라미터 CSV", data=csv_buf.getvalue(),
                     file_name=f"{meta['source_file']}_fit_params.csv",
                     mime='text/csv', use_container_width=True)

be_desc = be[::-1]; counts_desc = counts[::-1]
bg_desc = result['background'][::-1]
env_desc = (result['background'] + result['y_fit'])[::-1]
curves_dict = {
    'BE_eV': be_desc, 'Counts_exp': counts_desc,
    'Background': bg_desc, 'Envelope': env_desc,
}
for c in result['components']:
    curves_dict[c['label']] = (result['background'] + c['curve'])[::-1]
curves_df = pd.DataFrame(curves_dict)
curves_buf = io.StringIO()
curves_df.to_csv(curves_buf, index=False)
dl2.download_button("📊 피팅 곡선 CSV", data=curves_buf.getvalue(),
                     file_name=f"{meta['source_file']}_fit_curves.csv",
                     mime='text/csv', use_container_width=True)

png_buf = io.BytesIO()
fig.savefig(png_buf, format='png', dpi=180, bbox_inches='tight')
dl3.download_button("🖼️ 플롯 PNG", data=png_buf.getvalue(),
                     file_name=f"{meta['source_file']}_fit.png",
                     mime='image/png', use_container_width=True)


st.divider()
st.caption("XPS AutoFit v0.3 · *결과는 항상 도메인 지식으로 검증하세요.*")
