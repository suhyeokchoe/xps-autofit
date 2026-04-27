"""
XPS AutoFit — Streamlit Web App (v0.4)
================================================
v0.3 → v0.4:
- Expert 모드: 재료 템플릿 기반 제약 피팅
- Peak 위치/FWHM 공유 제약
- 논문 수준 피팅 (자유도 5~12 조절 가능)
- 정직한 결과 표시

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

st.set_page_config(
    page_title="XPS AutoFit",
    page_icon="📊",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        ### XPS AutoFit
        자동화된 XPS 피크 피팅 도구 · v0.4
        
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
        """
    }
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stMetric { background: #f0f2f6; padding: 0.8rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# 공통 헬퍼 (탭보다 먼저 정의)
# =========================================================================
def plot_result(result, meta, container, mode_label=None):
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
    total_with_bg = result['background'] + result['y_fit']
    ax_main.plot(be_p, total_with_bg, '-', color='red', lw=1.5,
                  label='Envelope', zorder=4)

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

    # 파라미터 테이블
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

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    dl1.download_button("📋 파라미터 CSV", data=csv_buf.getvalue(),
                         file_name=f"{meta['source_file']}_params.csv",
                         mime='text/csv', use_container_width=True,
                         key=f"dl_params_{mode_label}")

    be_desc = be_p[::-1]
    curves_dict = {
        'BE_eV': be_desc,
        'Counts_exp': counts_p[::-1],
        'Background': result['background'][::-1],
        'Envelope': (result['background'] + result['y_fit'])[::-1],
    }
    for c in result['components']:
        label = c.get('name', c.get('label', 'peak'))
        curves_dict[label] = (result['background'] + c['curve'])[::-1]
    curves_df = pd.DataFrame(curves_dict)
    curves_buf = io.StringIO()
    curves_df.to_csv(curves_buf, index=False)
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
# 제목
# =========================================================================
st.title("📊 XPS AutoFit")
st.caption("자동 XPS 피팅 · v0.4 · Expert 모드로 논문 수준 피팅 지원")


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


# =========================================================================
# 업로드 전 안내
# =========================================================================
if uploaded is None:
    st.info("👈 사이드바에서 XPS 데이터 파일을 업로드하세요.")
    with st.expander("🆕 v0.4 업데이트 — Expert 모드", expanded=True):
        st.markdown("""
**v0.4의 핵심: Expert 모드**

자동 감지만으로 피팅이 어려운 데이터(예: MOF O1s처럼 여러 화학결합이 겹친 경우)를
전문가 수준으로 피팅하는 새 기능입니다.

1. **재료 템플릿 선택** → MOF, Metal oxide, Polymer 등
2. **컴포넌트 자동 제안** → 해당 재료의 전형적인 피크들
3. **제약 옵션**: 위치 고정 / FWHM 공유 / η 공유
4. **정직한 결과**: 데이터가 지지하지 않는 피크는 면적이 작게 나옴

#### 지원 재료 템플릿
""")
        for k, v in MATERIAL_TEMPLATES.items():
            comps = ", ".join([c['name'] for c in v['components']])
            st.markdown(f"- **{k}** ({v['region']}): {comps}")
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

region_detected = meta.get('region', 'unknown')

col1, col2, col3, col4 = st.columns(4)
col1.metric("파일", meta['source_file'])
col2.metric("감지된 region", region_detected)
col3.metric("Points", len(be))
col4.metric("BE 범위", f"{be.max():.1f} → {be.min():.1f} eV")


# =========================================================================
# 모드 탭
# =========================================================================
tab_auto, tab_expert = st.tabs(["🤖 자동 모드", "🔬 Expert 모드"])


# -------------------------------------------------------------------
# 자동 모드
# -------------------------------------------------------------------
with tab_auto:
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
            result = auto_fit_v3(be, counts, eff_meta, max_peaks=max_peaks)
        else:
            bg = shirley_background(be, counts)
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
                'n_peaks': manual_n, 'trials': None,
                'doublet_info': None,
            }

    if not result['success']:
        st.error(f"피팅 실패: {result['reason']}")
    else:
        mode_info = f"{result['n_peaks']}개 " + (
            '상태(doublet)' if result['mode'] == 'doublet' else '피크')
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("모델", mode_info)
        m2.metric("R²", f"{result['r_squared']:.4f}")
        m3.metric("RMS", f"{result['rms']:.1f}")
        m4.metric("AIC", f"{result['aic']:.1f}")

        plot_result(result, meta, st, mode_label='auto')


# -------------------------------------------------------------------
# Expert 모드
# -------------------------------------------------------------------
with tab_expert:
    st.markdown("#### 🔬 Expert 모드 — 재료 기반 제약 피팅")
    st.caption("논문 수준의 피팅이 필요할 때. 재료 템플릿 선택 → 컴포넌트 편집 → 피팅.")

    # 템플릿 선택
    col_t1, col_t2 = st.columns([2, 3])
    with col_t1:
        filtered = {k: v for k, v in MATERIAL_TEMPLATES.items()
                     if v['region'] == region_detected}
        if not filtered:
            filtered = MATERIAL_TEMPLATES
        template_name = st.selectbox(
            "재료 템플릿",
            options=list(filtered.keys()),
            key='exp_template'
        )
    with col_t2:
        tmpl = MATERIAL_TEMPLATES[template_name]
        st.info(f"**{tmpl['description']}**  \n참조: *{tmpl['reference']}*")

    # 옵션 컴포넌트
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

    # 컴포넌트 편집
    st.markdown("#### 컴포넌트 상세 설정")
    st.caption("각 컴포넌트의 위치/FWHM 범위/고정 여부를 조정할 수 있습니다.")

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
                lock_pos = st.checkbox(
                    "위치 완전 고정", value=False,
                    key=f"lock_{ukey}",
                    help="체크하면 BE가 정확히 고정됨"
                )
            with cc2:
                be_tol = st.number_input(
                    "BE 이동 ± (eV)", value=float(c.be_tol),
                    min_value=0.0, max_value=2.0, step=0.05, format="%.2f",
                    key=f"tol_{ukey}", disabled=lock_pos
                )
                fwhm_min = st.number_input(
                    "FWHM 최소 (eV)", value=float(c.fwhm_min),
                    min_value=0.3, step=0.1, format="%.1f",
                    key=f"fmin_{ukey}"
                )
            with cc3:
                fwhm_max = st.number_input(
                    "FWHM 최대 (eV)", value=float(c.fwhm_max),
                    min_value=0.4, step=0.1, format="%.1f",
                    key=f"fmax_{ukey}"
                )

            edited_comps.append(ComponentSpec(
                name=c.name, be=be_center, be_tol=be_tol,
                fwhm_min=fwhm_min, fwhm_max=fwhm_max,
                lock_position=lock_pos
            ))

    # 전역 제약
    st.markdown("#### 전역 제약")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        share_fwhm = st.checkbox(
            "모든 컴포넌트 FWHM 공유", value=False,
            help="같은 재료의 모든 피크가 같은 FWHM을 가진다고 가정"
        )
    with gc2:
        share_eta = st.checkbox(
            "모든 컴포넌트 η 공유", value=False,
            help="모든 피크가 같은 Gauss-Lorentz 비율"
        )
    with gc3:
        use_shirley_exp = st.checkbox(
            "Shirley 배경 보정", value=True,
            help="논문 데이터처럼 배경이 이미 제거된 경우 해제"
        )

    # 자유 파라미터 추정
    n_params = 0
    for c in edited_comps:
        n_params += 1
        if not c.lock_position: n_params += 1
        if not share_fwhm: n_params += 1
        if not share_eta: n_params += 1
    if share_fwhm: n_params += 1
    if share_eta: n_params += 1
    st.caption(f"📊 자유 파라미터 ≈ **{n_params}개**  "
               f"(완전자유: {4*len(edited_comps)}, 최대제약: {len(edited_comps)+2})")

    # 피팅
    fit_btn = st.button("🎯 Expert 피팅 실행", type='primary',
                         use_container_width=True)

    if fit_btn:
        with st.spinner("제약 피팅 중..."):
            exp_result = expert_fit(
                be, counts, edited_comps,
                share_fwhm=share_fwhm, share_eta=share_eta,
                use_shirley=use_shirley_exp
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
                st.info(f"🔗 공유 FWHM: {exp_result['shared_fwhm_value']:.3f} eV  "
                         + (f"|  공유 η: {exp_result['shared_eta_value']:.3f}"
                            if exp_result.get('shared_eta_value') is not None else ""))

            plot_result(exp_result, meta, st, mode_label='expert')

            # 정직성 체크
            tiny = [c for c in exp_result['components'] if c['area_pct'] < 5]
            if tiny:
                names = ", ".join([f"{c['name']} ({c['area_pct']:.1f}%)" for c in tiny])
                st.warning(
                    f"⚠️ **정직성 체크**: {names} — 면적 5% 미만. "
                    f"데이터가 이 컴포넌트를 실제로 지지하지 않을 수 있습니다. "
                    f"제거하고 재피팅을 고려해보세요."
                )
# =========================================================================
# 푸터: 면책조항 + 데이터 정책 + 인용 안내
# =========================================================================
st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    with st.expander("⚖️ 이용 시 주의사항", expanded=False):
        st.markdown("""
자유 이용 정책 : 
이 도구는 공익 목적의 학술 도구입니다.
연구, 교육, 논문 작성, 학회 발표 등에 자유롭게 사용 가능합니다.

결과 검증 의무 : 
자동 피팅 결과는 통계적 최적해이며, 화학적·물리적 타당성은
사용자가 도메인 지식으로 직접 검증해야 합니다. 본 도구의 결과를
근거로 한 의사결정·출판물·산업적 응용에 대한 책임은 사용자에게 있습니다.

알려진 한계
- F1s, C1s, O1s 등 일반적 region에 최적화되어 있습니다.
- 비대칭 라인쉐입(LA, DS) 미지원합니다. 이용자가 많아지면 그때 고려해보겠습니다....
- 화학상태 자동 라벨링은 도메인 prior 기반 (절대 정답 아님!)
""")

with footer_col2:
    with st.expander("🔒 데이터 처리 정책", expanded=False):
        st.markdown("""
데이터 보호
- 업로드된 XPS 데이터는 서버 메모리에서만 처리됩니다.
- 세션 종료 시 데이터는 자동 삭제되며, 별도 저장소에 보관하지 않습니다.
- 어떤 형태의 사용자 식별정보도 수집하지 않습니다.

익명 통계
- Streamlit Cloud에서 익명 접속 통계가 자동 수집될 수 있으나,
  본 도구가 별도로 사용자 데이터를 추적하지 않습니다.

버그 / 기능 제안
- GitHub Issues 또는 'Manage app'을 통해 전달해주세요. 감사합니다.
""")

st.markdown(
    "<div style='text-align:center; color:#888; padding:1rem 0; font-size:0.85em;'>"
    "XPS AutoFit · v0.4 · 결과는 항상 도메인 지식으로 검증하세요."
    "</div>",
    unsafe_allow_html=True
)