"""
XPS AutoFit — Streamlit Web App (v0.5)
================================================
v0.4 → v0.5:
- Survey scan 자동 분석 추가
- 자동 모드에서 BE 범위로 Narrow/Survey 자동 분기
- 별도 Survey 탭 신설
- 원소 자동 식별 + atomic % 정량 (multi-line consistency check)

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
from xps_expert import (
    MATERIAL_TEMPLATES, ComponentSpec,
    components_from_template, expert_fit,
)
from xps_survey import (
    analyze_survey, is_survey_scan, ELEMENT_DB,
)
from xps_multimatch import (
    auto_match_templates, TemplateMatchResult, get_compatible_templates,
)

st.set_page_config(
    page_title="XPS AutoFit",
    page_icon="📊",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        ### XPS AutoFit
        자동화된 XPS 피크 피팅 도구 · v0.5

        **공익 목적의 학술 도구**로 개발되었습니다.
        연구·교육·논문 작성에 자유롭게 사용 가능하며,
        결과의 화학적·물리적 타당성은 사용자가 검증해주세요.

        **프로젝트 / 코드 / 기여자**
        - Repository: [github.com/suhyeokchoe/xps-autofit](https://github.com/suhyeokchoe/xps-autofit)
        - Authors: [AUTHORS.md](https://github.com/suhyeokchoe/xps-autofit/blob/main/AUTHORS.md)
        - License: MIT

        **참고 문헌**
        - CasaXPS Cookbook (Casa Software Ltd, 2019)
        - Shirley, D. A. (1972). *Phys. Rev. B*, 5(12), 4709
        - Akaike, H. (1974). *IEEE Trans. Auto. Control*, 19(6), 716
        - NIST X-ray Photoelectron Spectroscopy Database
        """
    }
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stMetric { background: #f0f2f6; padding: 0.8rem; border-radius: 0.5rem; }
    .conf-high { background: #06a77d; color: white; padding: 2px 8px;
                  border-radius: 4px; font-size: 0.85em; }
    .conf-medium { background: #f4a261; color: white; padding: 2px 8px;
                    border-radius: 4px; font-size: 0.85em; }
    .conf-low { background: #aaaaaa; color: white; padding: 2px 8px;
                 border-radius: 4px; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# 공통 헬퍼: Narrow 결과 플롯
# =========================================================================
def plot_narrow_result(result, meta, container, mode_label=None):
    be_p = result['be']; counts_p = result['counts']
    colors = ['#e63946', '#457b9d', '#06a77d', '#f4a261',
              '#9b5de5', '#f15bb5', '#00bbf9']

    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={'height_ratios': [4, 1]}, sharex=True
    )
    ax_main.plot(be_p, counts_p, 'o', mfc='none', mec='black', ms=3,
                  label='Experimental', zorder=3)
    ax_main.plot(be_p, result['background'], '--', color='gray', lw=1,
                  label='BG', zorder=2)
    ax_main.plot(be_p, result['background'] + result['y_fit'], '-',
                  color='red', lw=1.5, label='Envelope', zorder=4)

    for i, comp in enumerate(result['components']):
        color = colors[i % len(colors)]
        comp_full = result['background'] + comp['curve']
        ax_main.fill_between(be_p, result['background'], comp_full,
                              alpha=0.3, color=color)
        label_key = comp.get('name', comp.get('label', f'Peak {i+1}'))
        ax_main.plot(be_p, comp_full, '-', color=color, lw=1.2,
                      label=f"{label_key}: {comp['position']:.2f} eV "
                            f"({comp['area_pct']:.1f}%)")

    ax_main.set_ylabel('Counts / s')
    title_mode = mode_label or result.get('mode', '')
    ax_main.set_title(f"{meta['source_file']} — {meta.get('region','?')} ({title_mode})")
    ax_main.invert_xaxis()
    ax_main.legend(loc='upper right', fontsize=9)
    ax_main.grid(alpha=0.2)

    resid = counts_p - (result['background'] + result['y_fit'])
    ax_resid.plot(be_p, resid, 'o-', mfc='none', mec='gray',
                   color='gray', ms=2, lw=0.5)
    ax_resid.axhline(0, color='red', lw=0.8)
    ax_resid.set_xlabel('Binding Energy (eV)')
    ax_resid.set_ylabel('Residual')
    ax_resid.grid(alpha=0.2)

    plt.tight_layout()
    container.pyplot(fig)

    # 테이블
    container.subheader("🔢 컴포넌트 파라미터")
    rows = []
    for c in result['components']:
        label = c.get('name', c.get('label', ''))
        row = {
            'Component': label,
            'Position (eV)': round(c['position'], 3),
            'FWHM (eV)': round(c['fwhm'], 3),
            'Amplitude': round(c['amplitude'], 1),
            'η (GL ratio)': round(c['eta'], 3),
            'Area (%)': round(c['area_pct'], 2),
        }
        if 'be_err' in c and c['be_err'] > 0:
            row['BE err (±)'] = round(c['be_err'], 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    container.dataframe(df, use_container_width=True, hide_index=True)

    # 다운로드
    container.divider()
    container.subheader("💾 다운로드")
    dl1, dl2, dl3 = container.columns(3)

    csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
    dl1.download_button("📋 파라미터 CSV", data=csv_buf.getvalue(),
                         file_name=f"{meta['source_file']}_params.csv",
                         mime='text/csv', use_container_width=True,
                         key=f"dl_params_{mode_label}")

    be_desc = be_p[::-1]
    curves_dict = {
        'BE_eV': be_desc, 'Counts_exp': counts_p[::-1],
        'Background': result['background'][::-1],
        'Envelope': (result['background'] + result['y_fit'])[::-1],
    }
    for c in result['components']:
        label = c.get('name', c.get('label', 'peak'))
        curves_dict[label] = (result['background'] + c['curve'])[::-1]
    curves_df = pd.DataFrame(curves_dict)
    curves_buf = io.StringIO(); curves_df.to_csv(curves_buf, index=False)
    dl2.download_button("📊 피팅 곡선 CSV", data=curves_buf.getvalue(),
                         file_name=f"{meta['source_file']}_curves.csv",
                         mime='text/csv', use_container_width=True,
                         key=f"dl_curves_{mode_label}")

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format='png', dpi=180, bbox_inches='tight')
    dl3.download_button("🖼️ 플롯 PNG", data=png_buf.getvalue(),
                         file_name=f"{meta['source_file']}_fit.png",
                         mime='image/png', use_container_width=True,
                         key=f"dl_png_{mode_label}")


# =========================================================================
# 공통 헬퍼: Survey 결과 플롯
# =========================================================================
def plot_survey_result(result, meta, container):
    """Survey 분석 결과를 시각화 + 테이블 + 정량 + 다운로드"""
    be_p = result['be']; counts_p = result['counts']

    # 메트릭
    n_high = sum(1 for m in result['matches'] if m.confidence == 'high')
    n_med = sum(1 for m in result['matches'] if m.confidence == 'medium')
    n_low = sum(1 for m in result['matches'] if m.confidence == 'low')
    m1, m2, m3, m4 = container.columns(4)
    m1.metric("검출 피크", len(result['detected_peaks']))
    m2.metric("High confidence", n_high)
    m3.metric("Medium", n_med)
    m4.metric("Low (검증 필요)", n_low)

    # 시각화
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(be_p, counts_p, 'k-', lw=0.6)
    ax.invert_xaxis()
    ax.set_xlabel('Binding Energy (eV)')
    ax.set_ylabel('Counts / s')
    ax.set_title(f"Survey Auto-Analysis — {meta['source_file']}")

    conf_colors = {'high': '#06a77d', 'medium': '#f4a261', 'low': '#aaaaaa'}

    # 검출된 모든 피크에 라인
    for m in result['matches']:
        color = conf_colors[m.confidence]
        for line in m.matched_lines:
            ax.axvline(line['be_observed'], color=color, alpha=0.4, lw=0.6)

    # 주 피크에 라벨 (강도 큰 순으로 위에서 아래로 배치)
    sorted_matches = sorted(result['matches'],
                             key=lambda mm: -mm.primary_line_intensity)
    label_y_offset_factor = 1.0
    ymax = counts_p.max()
    for i, m in enumerate(sorted_matches):
        color = conf_colors[m.confidence]
        # 같은 BE에 라벨 겹치지 않도록 약간 변동
        y_offset = ymax * (0.95 - 0.05 * (i % 3))
        ax.annotate(
            f"{m.element}",
            xy=(m.primary_line_be, m.primary_line_intensity),
            xytext=(m.primary_line_be, y_offset),
            fontsize=10, ha='center', color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec=color, lw=0.8),
            arrowprops=dict(arrowstyle='-', color=color, alpha=0.5, lw=0.5)
        )

    ax.grid(alpha=0.2)
    plt.tight_layout()
    container.pyplot(fig)

    # 식별된 원소 테이블
    container.subheader("🧪 검출된 원소")
    rows = []
    for m in result['matches']:
        primary_shift = m.matched_lines[0]['be_shift']
        lines_str = ', '.join([f"{l['name']} @ {l['be_observed']:.1f}"
                                for l in m.matched_lines])
        rows.append({
            'Element': m.element,
            'Confidence': m.confidence.upper(),
            'Primary BE (eV)': round(m.primary_line_be, 2),
            'Charging shift (eV)': f"{primary_shift:+.1f}",
            'Matched lines': lines_str,
            '# lines': len(m.matched_lines),
        })
    df_elem = pd.DataFrame(rows)
    container.dataframe(df_elem, use_container_width=True, hide_index=True)

    # 신뢰도 안내
    with container.expander("ℹ️ Confidence 등급 설명"):
        st.markdown("""
- **HIGH** ★ Primary line + 2개 이상의 secondary line이 일관된 시프트로 매칭됨.
  → 매우 높은 신뢰도. 이 원소는 거의 확실히 샘플에 존재.
- **MEDIUM** Primary + 1개의 secondary line 매칭.
  → 합리적 신뢰도. 검증 위해 narrow scan 권장.
- **LOW** ⚠️ Primary line 1개만 매칭 (다른 원소의 line과 우연히 겹쳤을 수 있음).
  → **검증 필요**. 가짜 양성 가능성 있음. 다른 secondary line 위치를
  narrow scan으로 확인하거나, BE 범위 안에 secondary line이 있는지 점검하세요.
        """)

    # Atomic % 정량
    container.divider()
    container.subheader("📊 Approximate Atomic Composition")
    container.warning(
        "⚠️ **이 정량값은 근사치입니다.** "
        "Survey scan 기반 정량은 다음 한계를 가집니다:\n"
        "- 피크 면적이 아닌 강도(height)만 사용 (배경 보정 단순화)\n"
        "- Transmission function 보정 미적용\n"
        "- 같은 원소의 여러 라인 평균화 미적용\n\n"
        "**정확한 정량을 원하시면**: 각 원소의 narrow scan을 측정하고 "
        "면적 적분 + Shirley 배경 보정으로 다시 계산하세요. "
        "이 값은 **상대적 비율의 빠른 추정**으로만 활용하시기 바랍니다."
    )

    quant = result['quantification']
    if not quant or all(pct is None for _, pct in quant):
        container.info("정량 가능한 원소가 없습니다 (RSF 부재 또는 high/medium 매칭 부족).")
    else:
        col_pie, col_bar = container.columns([1, 1])

        # 파이 차트
        labels = []; sizes = []
        palette = ['#e63946', '#457b9d', '#06a77d', '#f4a261', '#9b5de5',
                    '#f15bb5', '#00bbf9', '#ffb703']
        colors = []
        for i, (m, pct) in enumerate(quant):
            if pct is not None:
                labels.append(f"{m.element}\n{pct:.1f}%")
                sizes.append(pct)
                colors.append(palette[i % len(palette)])
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        ax_pie.pie(sizes, labels=labels, colors=colors,
                    startangle=90, textprops={'fontsize': 11},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        ax_pie.set_title('Approximate Atomic %')
        col_pie.pyplot(fig_pie)

        # 정량 테이블
        rows_q = []
        for m, pct in quant:
            primary = m.matched_lines[0]
            rows_q.append({
                'Element': m.element,
                'Primary line': primary['name'],
                'BE (eV)': round(primary['be_observed'], 2),
                'Intensity': int(primary['intensity']),
                'RSF': primary.get('rsf', 0),
                'Atomic %': round(pct, 2) if pct is not None else 'N/A',
                'Confidence': m.confidence.upper(),
            })
        df_q = pd.DataFrame(rows_q)
        col_bar.dataframe(df_q, use_container_width=True, hide_index=True)

    # 다운로드
    container.divider()
    container.subheader("💾 다운로드")
    dl1, dl2, dl3 = container.columns(3)

    csv_buf = io.StringIO(); df_elem.to_csv(csv_buf, index=False)
    dl1.download_button("🧪 식별 원소 CSV", data=csv_buf.getvalue(),
                         file_name=f"{meta['source_file']}_elements.csv",
                         mime='text/csv', use_container_width=True,
                         key='dl_survey_elem')

    if quant:
        df_q_export = pd.DataFrame([{
            'Element': m.element,
            'Primary_line': m.matched_lines[0]['name'],
            'BE_eV': round(m.matched_lines[0]['be_observed'], 2),
            'Intensity': int(m.matched_lines[0]['intensity']),
            'RSF': m.matched_lines[0].get('rsf', 0),
            'Atomic_pct': round(pct, 2) if pct is not None else None,
            'Confidence': m.confidence,
        } for m, pct in quant])
        csv_q_buf = io.StringIO(); df_q_export.to_csv(csv_q_buf, index=False)
        dl2.download_button("📊 정량 CSV", data=csv_q_buf.getvalue(),
                             file_name=f"{meta['source_file']}_quantification.csv",
                             mime='text/csv', use_container_width=True,
                             key='dl_survey_quant')

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format='png', dpi=180, bbox_inches='tight')
    dl3.download_button("🖼️ 플롯 PNG", data=png_buf.getvalue(),
                         file_name=f"{meta['source_file']}_survey.png",
                         mime='image/png', use_container_width=True,
                         key='dl_survey_png')


# =========================================================================
# 제목
# =========================================================================
st.title("📊 XPS AutoFit")
st.caption("자동 XPS 피팅 · v0.5 · Survey scan 자동 분석 추가")


# =========================================================================
# 사이드바
# =========================================================================
with st.sidebar:
    st.header("📤 데이터 업로드")
    uploaded = st.file_uploader(
        "CSV 또는 TXT 파일", type=['csv', 'txt'],
        help="CasaXPS export 또는 2열(BE, Counts) CSV"
    )
    st.caption("💡 .xls는 Excel에서 CSV로 저장 후 업로드")

    st.divider()
    st.header("🎯 Calibration")
    apply_cal = st.checkbox("BE offset 적용", value=False)
    cal_shift = st.number_input(
        "Shift (eV)", value=0.0, step=0.05, format="%.2f",
        disabled=not apply_cal
    )

    st.divider()
    st.header("📐 Background Anchor")
    st.caption(
        "Shirley BG의 시작/끝 위치를 결정합니다. "
        "복잡한 데이터는 자동 감지가 부정확할 수 있어요."
    )
    bg_mode = st.radio(
        "BG 영역 결정 방식",
        ['Auto (피크 영역 자동 감지)',
         'Full range (양 끝점 그대로)',
         'Manual (BE 직접 지정)'],
        index=0,
        help=(
            "**Auto**: 1차 미분으로 피크 시작/끝을 자동 감지 (권장)\n\n"
            "**Full range**: 데이터 양 끝을 그대로 anchor로 사용 "
            "(끝부분에 피크가 있으면 BG가 부정확)\n\n"
            "**Manual**: 사용자가 BE 값으로 직접 지정"
        )
    )

    bg_anchor_left = None
    bg_anchor_right = None
    if bg_mode == 'Manual (BE 직접 지정)':
        be_min_val = float(0)  # 나중에 데이터 로드 후 갱신
        bg_anchor_left = st.number_input(
            "낮은 BE 쪽 anchor (eV)",
            value=0.0, step=0.5, format="%.2f",
            help="BG가 anchor BE 값에서 raw data를 따라가도록 함"
        )
        bg_anchor_right = st.number_input(
            "높은 BE 쪽 anchor (eV)",
            value=0.0, step=0.5, format="%.2f"
        )


# =========================================================================
# 업로드 전
# =========================================================================
if uploaded is None:
    st.info("👈 사이드바에서 XPS 데이터 파일을 업로드하세요.")
    with st.expander("🆕 v0.5 업데이트 — Survey 자동 분석", expanded=True):
        st.markdown("""
**v0.5의 핵심: Survey scan 자동 분석**

지금까지는 narrow scan만 다뤘는데, 이제 **survey scan**도 자동 처리합니다:
- **자동 모드**: 데이터 BE 범위가 500 eV 이상이면 자동으로 Survey 분석으로 분기
- **별도 Survey 탭**: 명시적으로 Survey 분석을 원할 때

**Survey 분석 기능**:
1. 피크 자동 검출 (1차 미분 + prominence)
2. **다중 라인 자기일관성 매칭** — 단일 피크 매칭의 가짜 양성 방지
3. 충전(charging) 시프트 자동 처리
4. 신뢰도 등급 (HIGH / MEDIUM / LOW)
5. **근사 atomic %** 정량 (Scofield RSF 기반)

지원 원소: Li, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca,
Ti, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, As, Mo, Ag, In, Sn, Hf, Ta, W, Pt, Au
        """)
    with st.expander("🔬 v0.4 — Expert 모드 (Narrow)"):
        st.markdown("""
- 재료 템플릿 기반 제약 피팅
- 위치 고정 / FWHM 공유 / η 공유 옵션
- 정직한 결과 (데이터가 지지하지 않는 컴포넌트는 면적 작게)
        """)
    st.stop()


# =========================================================================
# 데이터 로딩
# =========================================================================
try:
    raw_bytes = uploaded.read()
    text = raw_bytes.decode('utf-8-sig', errors='replace')
    be, counts, meta = load_xps_csv(text, source_name=uploaded.name)
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

if apply_cal and cal_shift != 0:
    be = calibrate_shift(be, cal_shift)
    meta['calibrated'] = f"shift={cal_shift:+.2f} eV"

# BG 모드에 따라 옵션 dict 생성
bg_kwargs = {}
if bg_mode == 'Auto (피크 영역 자동 감지)':
    bg_kwargs = {'auto_anchor': True}
elif bg_mode == 'Full range (양 끝점 그대로)':
    bg_kwargs = {'auto_anchor': False}
elif bg_mode == 'Manual (BE 직접 지정)':
    bg_kwargs = {
        'anchor_left': bg_anchor_left,
        'anchor_right': bg_anchor_right,
    }

# 자동 분기 결정
auto_is_survey = is_survey_scan(be)
region_detected = meta.get('region', 'unknown')

col1, col2, col3, col4 = st.columns(4)
col1.metric("파일", meta['source_file'])
col2.metric("타입", "Survey" if auto_is_survey else "Narrow")
col3.metric("Points", len(be))
col4.metric("BE 범위", f"{be.max():.1f} → {be.min():.1f} eV")


# =========================================================================
# 모드 탭
# =========================================================================
tab_auto, tab_expert, tab_survey = st.tabs(
    ["🤖 자동 모드", "🔬 Expert 모드 (Narrow)", "🌐 Survey 분석"]
)


# -------------------------------------------------------------------
# 자동 모드 — 자동 분기
# -------------------------------------------------------------------
with tab_auto:
    if auto_is_survey:
        st.success(
            "📡 **Survey scan으로 자동 감지되어 원소 자동 식별 모드로 전환합니다.** "
            "(BE 범위 500 eV 이상)"
        )
        with st.spinner("Survey 분석 중..."):
            result = analyze_survey(be, counts)

        if not result['success']:
            st.error(f"분석 실패: {result['reason']}")
        else:
            plot_survey_result(result, meta, st)

    else:
        # ---- Narrow 자동 모드 (v0.7 단순 방식) ----
        st.markdown("AIC 기반 자동 피크 개수 선택 + singlet/doublet 자동 분기")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            all_regions = ['auto'] + list(ELEMENT_PRIORS.keys()) + list(DOUBLET_PRIORS.keys())
            region_override = st.selectbox("Region", all_regions, index=0)
        with col_b:
            max_peaks = st.slider("최대 탐색 피크 수", 1, 6, 3)
        with col_c:
            manual_n = st.number_input("수동 지정 (0=자동)", 0, 6, 0)

        force_singlet = st.checkbox("Doublet 강제 해제 (singlet만)", value=False)

        eff_meta = dict(meta)
        if region_override != 'auto':
            eff_meta['region'] = region_override
        if force_singlet and eff_meta.get('region') in DOUBLET_PRIORS:
            eff_meta['region'] = 'unknown'

        with st.spinner("피팅 중..."):
            if manual_n == 0:
                result = auto_fit_v3(be, counts, eff_meta,
                                       max_peaks=max_peaks,
                                       bg_kwargs=bg_kwargs)
            else:
                bg = shirley_background(be, counts, **bg_kwargs)
                y_corr = counts - bg
                peaks_idx, _ = detect_peaks_v2(be, y_corr, eff_meta.get('region'))
                if len(peaks_idx) == 0:
                    st.error("피크 감지 실패"); st.stop()
                init_centers = sorted([float(be[i]) for i in peaks_idx])
                ranked = sorted(init_centers,
                                key=lambda c: -y_corr[int(np.argmin(np.abs(be - c)))])
                centers = sorted(ranked[:manual_n])
                while len(centers) < manual_n:
                    centers.append(centers[-1] - 1.5 if centers else float(be[np.argmax(y_corr)]))
                if is_doublet(eff_meta.get('region')) and not force_singlet:
                    fit = fit_n_doublets(be, y_corr, manual_n, centers, eff_meta['region'])
                    _, _, _, _, dBE, ar = DOUBLET_PRIORS[eff_meta['region']]
                    components = []
                    for i in range(manual_n):
                        amp_m, c_m, fwhm, eta = fit['popt'][i*4:i*4+4]
                        cm = pseudo_voigt(be, amp_m, c_m, fwhm, eta)
                        components.append({
                            'amplitude': float(amp_m), 'position': float(c_m),
                            'fwhm': float(fwhm), 'eta': float(eta),
                            'area': float(abs(np.trapezoid(cm, be))),
                            'curve': cm, 'label': f'State {i+1} (main)'})
                        cn = pseudo_voigt(be, amp_m/ar, c_m+dBE, fwhm, eta)
                        components.append({
                            'amplitude': float(amp_m/ar), 'position': float(c_m+dBE),
                            'fwhm': float(fwhm), 'eta': float(eta),
                            'area': float(abs(np.trapezoid(cn, be))),
                            'curve': cn, 'label': f'State {i+1} (minor)'})
                else:
                    fit = fit_n_peaks(be, y_corr, manual_n, centers, eff_meta.get('region'))
                    if fit is None:
                        st.error(f"{manual_n}개 피팅 실패"); st.stop()
                    components = []
                    for i in range(manual_n):
                        a, c, f, e = fit['popt'][i*4:i*4+4]
                        comp = pseudo_voigt(be, a, c, f, e)
                        components.append({
                            'amplitude': float(a), 'position': float(c),
                            'fwhm': float(f), 'eta': float(e),
                            'area': float(abs(np.trapezoid(comp, be))),
                            'curve': comp, 'label': f'Peak {i+1}'})
                components.sort(key=lambda c: -c['position'])
                for i, c in enumerate(components):
                    if c['label'].startswith('Peak '):
                        c['label'] = f'Peak {i+1}'
                tot = sum(c['area'] for c in components) or 1
                for c in components:
                    c['area_pct'] = 100 * c['area'] / tot
                result = {
                    'success': True,
                    'mode': 'doublet' if is_doublet(eff_meta.get('region')) and not force_singlet else 'singlet',
                    'be': be, 'counts': counts, 'background': bg,
                    'y_fit': fit['y_fit'], 'components': components,
                    'r_squared': fit['r2'], 'rms': fit['rms'], 'aic': fit['aic'],
                    'n_peaks': manual_n, 'trials': None, 'doublet_info': None,
                }

        if not result['success']:
            st.error(f"피팅 실패: {result['reason']}")
        else:
            mode_info = (f"{result['n_peaks']}개 " +
                          ('상태(doublet)' if result['mode'] == 'doublet' else '피크'))
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("모델", mode_info)
            mc2.metric("R²", f"{result['r_squared']:.4f}")
            mc3.metric("RMS", f"{result['rms']:.1f}")
            mc4.metric("AIC", f"{result['aic']:.1f}")
            plot_narrow_result(result, meta, st, mode_label='auto')




# -------------------------------------------------------------------
# Expert 모드 (v0.4와 동일, narrow 전용)
# -------------------------------------------------------------------
with tab_expert:
    if auto_is_survey:
        st.warning(
            "⚠️ 업로드한 데이터는 Survey scan으로 보입니다 (BE 범위 500 eV 이상). "
            "Expert 모드는 narrow scan용입니다. **Survey 분석** 탭을 사용하세요."
        )
    st.markdown("#### 🔬 Expert 모드 — 재료 기반 제약 피팅 (Narrow scan용)")
    st.caption("논문 수준의 피팅. 재료 템플릿 선택 → 컴포넌트 편집 → 피팅.")

    # ===========================================================
    # 🤖 자동 제안 섹션 (선택, 접혀있는 상태로 시작)
    # ===========================================================
    with st.expander("🤖 자동 제안 받기 — 어떤 템플릿이 좋을지 모르겠다면",
                       expanded=False):
        st.caption(
            "현재 데이터를 모든 호환 템플릿에 자동 매칭하고 R² 순으로 보여줍니다. "
            "원하는 카드를 선택하면 아래 **재료 템플릿**이 그것으로 자동 전환됩니다."
        )

        # region 결정 (자동 감지 또는 사용자가 위에서 정한 것)
        sugg_region = None
        if region_detected and region_detected != 'unknown':
            sugg_region = region_detected
        else:
            # BE 범위로 추정
            be_center = (be.max() + be.min()) / 2
            for r_name, (lo, hi) in [
                ('F1s', (680, 695)), ('O1s', (525, 540)),
                ('C1s', (280, 295)), ('N1s', (395, 410)),
            ]:
                if lo - 5 <= be_center <= hi + 5:
                    sugg_region = r_name
                    break

        if not sugg_region:
            st.warning(
                "Region을 추정할 수 없습니다. 아래 **재료 템플릿**에서 직접 선택해주세요."
            )
        else:
            run_match = st.button(
                f"🔍 {sugg_region} 템플릿들 자동 매칭하기",
                use_container_width=True,
                key='run_auto_match'
            )

            # 한 번 누르면 결과를 session_state에 저장 (탭 전환 시 유지)
            if run_match:
                with st.spinner(f"{sugg_region} region의 모든 템플릿 매칭 중..."):
                    match_results = auto_match_templates(
                        be, counts, region_hint=sugg_region,
                        bg_kwargs=bg_kwargs, max_results=6
                    )
                st.session_state['exp_match_results'] = match_results
                st.session_state['exp_match_region'] = sugg_region

            # 저장된 결과가 있으면 표시
            saved_results = st.session_state.get('exp_match_results')
            if saved_results and st.session_state.get('exp_match_region') == sugg_region:
                st.success(f"✓ {len(saved_results)}개 모델 매칭 완료. "
                            "원하는 결과의 **'이 템플릿 적용'** 버튼을 누르세요.")

                # 카드 표시 (3개씩 한 줄)
                label_color = {
                    'best': '🟢', 'good': '🟢',
                    'acceptable': '🟡', 'poor': '🔴'
                }
                for row_start in range(0, len(saved_results), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        idx = row_start + j
                        if idx >= len(saved_results):
                            continue
                        r = saved_results[idx]
                        with col:
                            marker = label_color.get(r.label, '⚪')
                            star = '⭐ ' if r.rank == 1 else ''
                            with st.container(border=True):
                                # 카드 제목 (free position 표시는 짧게)
                                disp_name = r.template_name.replace(
                                    ' [free position]', ' (자유위치)'
                                )
                                st.markdown(f"**{star}{marker} {disp_name}**")
                                cm1, cm2 = st.columns(2)
                                cm1.metric("R²", f"{r.r_squared:.4f}")
                                cm2.metric("자유도", r.n_free_params)
                                st.caption(f"{r.n_components} components · "
                                             f"{r.label}")
                                # "적용" 버튼 → 그 템플릿 이름을 session_state에
                                if st.button(
                                    "이 템플릿 적용",
                                    key=f"apply_template_{idx}",
                                    use_container_width=True,
                                ):
                                    # Statistical auto는 Expert 템플릿이 아니라 적용 불가
                                    if r.template_name.startswith('Statistical'):
                                        st.warning(
                                            "⚠️ Statistical auto는 통계 자동 모드입니다. "
                                            "Expert에 적용할 수 없습니다. "
                                            "자동 모드 탭에서 사용하세요."
                                        )
                                    else:
                                        # 'XXX [free position]' → 'XXX'로 정규화
                                        clean_name = r.template_name.replace(
                                            ' [free position]', ''
                                        )
                                        st.session_state['exp_template'] = clean_name
                                        st.session_state['exp_template_applied'] = clean_name
                                        st.rerun()

    # 적용된 템플릿 표시
    applied_msg = st.session_state.get('exp_template_applied')
    if applied_msg:
        st.info(f"✓ **자동 제안에서 적용됨**: `{applied_msg}` "
                  "— 아래에서 컴포넌트를 미세조정 후 피팅하세요.")

    # ===========================================================
    # 재료 템플릿 선택 + 편집 (기존 + session_state 연결)
    # ===========================================================
    col_t1, col_t2 = st.columns([2, 3])
    with col_t1:
        filtered = {k: v for k, v in MATERIAL_TEMPLATES.items()
                     if v['region'] == region_detected}
        if not filtered:
            filtered = MATERIAL_TEMPLATES
        # session_state에 자동 제안으로 적용된 게 있으면 default로 사용
        default_idx = 0
        applied = st.session_state.get('exp_template')
        if applied and applied in filtered:
            default_idx = list(filtered.keys()).index(applied)
        template_name = st.selectbox(
            "재료 템플릿", options=list(filtered.keys()),
            index=default_idx, key='exp_template_select'
        )
    with col_t2:
        tmpl = MATERIAL_TEMPLATES[template_name]
        st.info(f"**{tmpl['description']}**  \n참조: *{tmpl['reference']}*")

    selected_optional = []
    if tmpl.get('optional_components'):
        st.markdown("**옵션 컴포넌트**")
        for opt in tmpl['optional_components']:
            key = f"opt_{template_name}_{opt['name']}"
            checked = st.checkbox(
                f"{opt['name']} ({opt['be']} eV) — {opt.get('hint', '')}",
                key=key
            )
            if checked:
                selected_optional.append(opt['name'])

    base_comps = components_from_template(template_name, selected_optional)

    st.markdown("#### 컴포넌트 상세 설정")
    edited_comps = []
    for i, c in enumerate(base_comps):
        with st.expander(f"**{c.name}** @ {c.be} eV", expanded=False):
            cc1, cc2, cc3 = st.columns(3)
            ukey = f"{template_name}_{c.name}_{i}"
            with cc1:
                be_center = st.number_input(
                    "BE 중심 (eV)", value=float(c.be), step=0.05, format="%.2f",
                    key=f"be_{ukey}"
                )
                lock_pos = st.checkbox("위치 완전 고정", value=False,
                                         key=f"lock_{ukey}")
            with cc2:
                be_tol = st.number_input(
                    "BE 이동 ± (eV)", value=float(c.be_tol),
                    min_value=0.0, max_value=2.0, step=0.05, format="%.2f",
                    key=f"tol_{ukey}", disabled=lock_pos
                )
                fwhm_min = st.number_input(
                    "FWHM 최소 (eV)", value=float(c.fwhm_min),
                    min_value=0.3, step=0.1, format="%.1f", key=f"fmin_{ukey}"
                )
            with cc3:
                fwhm_max = st.number_input(
                    "FWHM 최대 (eV)", value=float(c.fwhm_max),
                    min_value=0.4, step=0.1, format="%.1f", key=f"fmax_{ukey}"
                )
            edited_comps.append(ComponentSpec(
                name=c.name, be=be_center, be_tol=be_tol,
                fwhm_min=fwhm_min, fwhm_max=fwhm_max,
                lock_position=lock_pos
            ))

    st.markdown("#### 전역 제약")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        share_fwhm = st.checkbox("모든 컴포넌트 FWHM 공유", value=False)
    with gc2:
        share_eta = st.checkbox("모든 컴포넌트 η 공유", value=False)
    with gc3:
        use_shirley_exp = st.checkbox("Shirley 배경 보정", value=True)

    n_params = sum(1 + (0 if c.lock_position else 1)
                    + (0 if share_fwhm else 1)
                    + (0 if share_eta else 1) for c in edited_comps)
    n_params += (1 if share_fwhm else 0) + (1 if share_eta else 0)
    st.caption(f"📊 자유 파라미터 ≈ **{n_params}개**")

    fit_btn = st.button("🎯 Expert 피팅 실행", type='primary',
                          use_container_width=True)

    if fit_btn:
        with st.spinner("제약 피팅 중..."):
            exp_result = expert_fit(
                be, counts, edited_comps,
                share_fwhm=share_fwhm, share_eta=share_eta,
                use_shirley=use_shirley_exp,
                bg_kwargs=bg_kwargs,
            )
        if not exp_result['success']:
            st.error(f"피팅 실패: {exp_result['reason']}")
        else:
            em1, em2, em3, em4 = st.columns(4)
            em1.metric("컴포넌트 수", exp_result['n_components'])
            em2.metric("R²", f"{exp_result['r_squared']:.4f}")
            em3.metric("자유 파라미터", exp_result['n_free_params'])
            em4.metric("AIC", f"{exp_result['aic']:.1f}")
            if exp_result.get('shared_fwhm_value'):
                st.info(f"🔗 공유 FWHM: {exp_result['shared_fwhm_value']:.3f} eV")
            plot_narrow_result(exp_result, meta, st, mode_label='expert')

            tiny = [c for c in exp_result['components'] if c['area_pct'] < 5]
            if tiny:
                names = ", ".join([f"{c['name']} ({c['area_pct']:.1f}%)" for c in tiny])
                st.warning(
                    f"⚠️ **정직성 체크**: {names} — 면적 5% 미만. "
                    f"데이터가 이 컴포넌트를 실제로 지지하지 않을 수 있습니다."
                )


# -------------------------------------------------------------------
# Survey 탭 (명시적)
# -------------------------------------------------------------------
with tab_survey:
    if not auto_is_survey:
        st.warning(
            "⚠️ 업로드한 데이터는 Narrow scan으로 보입니다 (BE 범위 500 eV 미만). "
            "Survey 분석은 일반적으로 0~1400 eV 정도의 wide scan에 적합합니다. "
            "그래도 아래에서 분석을 시도할 수 있지만, 결과가 의미 없을 수 있습니다."
        )

    st.markdown("#### 🌐 Survey 분석 — 원소 자동 식별 + 근사 정량")
    st.caption(
        "다중 라인 자기일관성 검증으로 원소를 식별합니다. "
        "충전(charging) 시프트는 자동으로 흡수됩니다."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        tolerance = st.slider(
            "Primary line tolerance (eV)", 1.0, 8.0, 4.0, 0.5,
            help="검출 피크가 DB 위치에서 얼마나 벗어나도 매칭으로 인정할지. "
                 "충전이 큰 샘플은 6~8 eV로."
        )
    with col_s2:
        min_prominence = st.slider(
            "피크 검출 민감도 (%)", 1.0, 10.0, 2.0, 0.5,
            help="작을수록 약한 피크도 잡힘. 노이즈가 많으면 4~5%로 올리세요."
        ) / 100.0

    run_survey = st.button("🌐 Survey 분석 실행", type='primary',
                             use_container_width=True, key='run_survey_btn')

    if run_survey or auto_is_survey:
        with st.spinner("Survey 분석 중..."):
            # 임시: prominence 조절을 위해 직접 호출
            from xps_survey import detect_survey_peaks, identify_elements, quantify_atomic_percent
            detected, bg = detect_survey_peaks(be, counts,
                                                  prominence_ratio=min_prominence)
            if not detected:
                st.error("피크 감지 실패. 검출 민감도를 낮춰보세요.")
            else:
                matches = identify_elements(detected, (be.min(), be.max()),
                                              tolerance_ev=tolerance)
                quant = quantify_atomic_percent(matches,
                                                  only_high_confidence=True)
                survey_result = {
                    'success': True, 'mode': 'survey',
                    'be': be, 'counts': counts, 'background': bg,
                    'detected_peaks': detected,
                    'matches': matches,
                    'quantification': quant,
                    'n_elements': len([m for m in matches if m.confidence != 'low']),
                }
                plot_survey_result(survey_result, meta, st)


# =========================================================================
# 푸터: 면책조항 + 데이터 정책 (v0.4 그대로)
# =========================================================================
st.divider()
fc1, fc2 = st.columns(2)
with fc1:
    with st.expander("⚖️ 이용 시 주의사항", expanded=False):
        st.markdown("""
**자유 이용 정책**
공익 목적의 학술 도구입니다. 연구·교육·논문에 자유롭게 사용 가능합니다.

**결과 검증 의무**
자동 결과는 통계적 최적해입니다. 화학적·물리적 타당성은 사용자가 도메인
지식으로 검증해야 합니다.

**알려진 한계**
- Survey 정량은 근사치 (Scofield RSF 기반, 면적이 아닌 강도)
- 비대칭 라인쉐입(LA, DS) 미지원
- 화학상태 자동 라벨링은 도메인 prior 기반
""")
with fc2:
    with st.expander("🔒 데이터 처리 정책", expanded=False):
        st.markdown("""
**데이터 보호**
- 업로드된 데이터는 서버 메모리에서만 처리됩니다.
- 세션 종료 시 데이터는 자동 삭제됩니다.
- 사용자 식별정보를 수집하지 않습니다.
""")

st.markdown(
    "<div style='text-align:center; color:#888; padding:1rem 0; font-size:0.85em;'>"
    "XPS AutoFit · v0.5 · 결과는 항상 도메인 지식으로 검증하세요."
    "</div>",
    unsafe_allow_html=True
)
