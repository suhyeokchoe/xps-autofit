"""
XPS Multi-Template Auto-Match Engine (v0.8)
==================================================
사용자가 데이터를 올리면, BE 범위와 일치하는 모든 재료 템플릿을 자동으로
시도하고 R² 기준으로 정렬해 카드 형태로 보여주기 위한 엔진.

설계 철학:
- 사용자에게 "이건 무슨 재료입니까?" 묻지 않음
- 대신 라이브러리 안의 모든 합리적 후보를 시도
- 사용자는 결과 N개를 한눈에 보고 선택
- 통계 기반 (R² + AIC) + 화학 라벨 함께 제시 → 정직성

각 후보 모델:
- Quick: Expert 4-peak 강제 변형 (위치 ±tol, FWHM 공유)
- Strict: 위치 완전 고정 + FWHM 공유 (논문 수준)
- Free: Expert 위치 자유 (R² 최고)
- Auto-statistical: 도메인 지식 없는 v0.7 자동 (baseline 비교용)
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from xps_expert import (
    MATERIAL_TEMPLATES, ComponentSpec,
    components_from_template, expert_fit,
)
from xps_engine import auto_fit_v3


@dataclass
class TemplateMatchResult:
    """단일 템플릿 매칭 결과"""
    template_name: str
    description: str
    region: str
    fit_result: dict           # expert_fit이 리턴한 dict
    r_squared: float
    aic: float
    n_components: int
    n_free_params: int
    rank: int = 0              # 정렬 후 순위 (1부터)
    label: str = ''            # 'best' / 'good' / 'fallback' 등


def auto_match_templates(
    be: np.ndarray,
    counts: np.ndarray,
    region_hint: Optional[str] = None,
    bg_kwargs: Optional[dict] = None,
    max_results: int = 5,
) -> List[TemplateMatchResult]:
    """
    사용자 데이터를 모든 적용 가능한 템플릿에 매칭.

    Args:
        be, counts: 데이터
        region_hint: 'O1s', 'C1s' 등 (None이면 BE 범위로 자동 추정)
        bg_kwargs: shirley_background에 전달할 옵션
        max_results: 최대 결과 수

    Returns:
        TemplateMatchResult 리스트, R² 내림차순 정렬
    """
    bg_kwargs = bg_kwargs or {}

    # 1) Region 자동 추정 (hint 없으면)
    if region_hint is None:
        region_hint = _infer_region_from_be(be)

    # 2) 해당 region의 모든 템플릿 추출
    candidate_templates = [
        (name, tmpl) for name, tmpl in MATERIAL_TEMPLATES.items()
        if tmpl['region'] == region_hint
    ]

    if not candidate_templates:
        return []

    # 3) 각 템플릿에 대해 "Strict" 변형으로 피팅 시도
    #    Strict: 위치 고정 + FWHM 공유 (가장 보수적, 화학적으로 안정)
    results = []
    for tmpl_name, tmpl in candidate_templates:
        try:
            comps = components_from_template(tmpl_name)
            # 위치 고정 모드로 시도
            for c in comps:
                c.lock_position = True
            fit = expert_fit(
                be, counts, comps,
                share_fwhm=True, share_eta=True,
                use_shirley=True, bg_kwargs=bg_kwargs
            )
            if fit['success']:
                results.append(TemplateMatchResult(
                    template_name=tmpl_name,
                    description=tmpl['description'],
                    region=tmpl['region'],
                    fit_result=fit,
                    r_squared=fit['r_squared'],
                    aic=fit['aic'],
                    n_components=fit['n_components'],
                    n_free_params=fit['n_free_params'],
                ))
        except Exception:
            continue

    # 4) "Free" 변형도 시도 (위치 자유, FWHM 공유) — 가장 R² 높을 가능성
    for tmpl_name, tmpl in candidate_templates:
        try:
            comps = components_from_template(tmpl_name)
            # 위치 자유, FWHM 공유
            fit = expert_fit(
                be, counts, comps,
                share_fwhm=True, share_eta=False,
                use_shirley=True, bg_kwargs=bg_kwargs
            )
            if fit['success']:
                results.append(TemplateMatchResult(
                    template_name=f"{tmpl_name} [free position]",
                    description=tmpl['description'] + ' (위치 자유)',
                    region=tmpl['region'],
                    fit_result=fit,
                    r_squared=fit['r_squared'],
                    aic=fit['aic'],
                    n_components=fit['n_components'],
                    n_free_params=fit['n_free_params'],
                ))
        except Exception:
            continue

    # 5) 통계 기반 자동 (v0.7 자동 모드) — baseline 비교용
    try:
        meta = {'region': region_hint}
        auto_result = auto_fit_v3(be, counts, meta, max_peaks=4, bg_kwargs=bg_kwargs)
        if auto_result.get('success'):
            # auto_fit_v3 결과를 expert_fit 결과 형식으로 어댑팅
            adapted = {
                'success': True,
                'be': auto_result['be'], 'counts': auto_result['counts'],
                'background': auto_result['background'],
                'y_corrected': auto_result['counts'] - auto_result['background'],
                'y_fit': auto_result['y_fit'],
                'components': auto_result['components'],
                'r_squared': auto_result['r_squared'],
                'rms': auto_result['rms'],
                'aic': auto_result['aic'],
                'n_components': auto_result.get('n_peaks', len(auto_result['components'])),
                'n_free_params': auto_result.get('n_peaks', 1) * 4,
            }
            results.append(TemplateMatchResult(
                template_name='Statistical auto (도메인 지식 없음)',
                description='AIC 기반 자동 피크 개수 결정 (재료 가정 없음)',
                region=region_hint,
                fit_result=adapted,
                r_squared=adapted['r_squared'],
                aic=adapted['aic'],
                n_components=adapted['n_components'],
                n_free_params=adapted['n_free_params'],
            ))
    except Exception:
        pass

    # 6) 정렬: R² 내림차순 (높은 게 위)
    results.sort(key=lambda r: -r.r_squared)

    # 7) 상위 max_results만 + 라벨 부여
    results = results[:max_results]
    for i, r in enumerate(results):
        r.rank = i + 1
        if i == 0:
            r.label = 'best'
        elif r.r_squared >= 0.99:
            r.label = 'good'
        elif r.r_squared >= 0.95:
            r.label = 'acceptable'
        else:
            r.label = 'poor'

    return results


def _infer_region_from_be(be: np.ndarray) -> str:
    """BE 범위로 region 추정"""
    be_min, be_max = float(be.min()), float(be.max())
    be_center = (be_min + be_max) / 2

    # 각 region의 일반 범위
    regions = {
        'F1s':  (680, 695),
        'O1s':  (525, 540),
        'C1s':  (280, 295),
        'N1s':  (395, 410),
    }
    # center가 가장 가까운 region 선택
    best_region = None
    best_dist = float('inf')
    for name, (lo, hi) in regions.items():
        center = (lo + hi) / 2
        dist = abs(be_center - center)
        if dist < best_dist and lo - 5 <= be_center <= hi + 5:
            best_dist = dist
            best_region = name
    return best_region or 'unknown'


def get_compatible_templates(region: str) -> List[str]:
    """주어진 region에 호환되는 템플릿 이름 리스트"""
    return [name for name, tmpl in MATERIAL_TEMPLATES.items()
            if tmpl['region'] == region]
