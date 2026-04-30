"""
XPS AutoFit — Expert Fitting Module (v0.4)
==============================================
화학상태 라이브러리 + 제약 피팅.

설계 철학:
1. 사용자가 재료 타입을 선택하면 컴포넌트가 자동 제안됨.
2. 사용자는 컴포넌트를 편집 (추가/삭제/위치 조정)할 수 있음.
3. 제약 플래그: 위치 고정, FWHM 공유, GL ratio 공유.
4. 결과는 "논문을 재현"이 아닌 "제약 하에서 가장 좋은 해"를 리턴.
5. Zr-O-Zr이 0.2%면 그대로 0.2%를 내보내고, 사용자가 판단.
"""
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# -------------------------------------------------------------------
# 재료 템플릿 라이브러리
# -------------------------------------------------------------------
# 각 컴포넌트:
#   name, BE_center, BE_tolerance (위치 이동 허용 범위),
#   fwhm_range (min, max)
# 문헌 기반 (CasaXPS cookbook, NIST XPS DB, 관련 논문들)
# -------------------------------------------------------------------

MATERIAL_TEMPLATES = {
    # ============================================================
    # O1s region (525-540 eV)
    # ============================================================
    'MOF (Zr-based)': {
        'region': 'O1s',
        'description': 'Zr-based MOFs: UiO-66/67, MOF-801/867',
        'reference': 'Nat. Commun. 16, 162 (2025); Wang et al. JMCA 2017',
        'components': [
            {'name': 'Zr-O-Zr (oxo cluster)', 'be': 529.4, 'be_tol': 0.3, 'fwhm': (0.9, 2.0)},
            {'name': 'Zr-O-C (carboxylate linker)', 'be': 530.7, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
            {'name': 'Zr-O-H (hydroxyl / adsorbed water)', 'be': 532.1, 'be_tol': 0.2, 'fwhm': (1.0, 2.2)},
        ],
        'optional_components': [
            {'name': 'S=O (sulfonate group)', 'be': 531.6, 'be_tol': 0.3, 'fwhm': (1.0, 1.8),
             'hint': '예: sulfate, sulfonate, TFSI/TFS 이온성 액체, 황 함유 분자'},
            {'name': 'C=O (carbonyl)', 'be': 531.3, 'be_tol': 0.3, 'fwhm': (1.0, 2.0),
             'hint': '예: carbonyl, ketone, aldehyde'},
            {'name': 'O-C=O (carboxylate organic)', 'be': 533.5, 'be_tol': 0.3, 'fwhm': (1.0, 2.0),
             'hint': '예: 자유 carboxylic acid (linker가 아닌)'},
        ],
    },
    'Metal oxide (generic)': {
        'region': 'O1s',
        'description': 'Generic transition metal oxide (TiO2, Fe2O3, SnO2, ZnO 등)',
        'reference': 'NIST XPS DB',
        'components': [
            {'name': 'M-O (lattice oxygen)',  'be': 530.0, 'be_tol': 0.5, 'fwhm': (1.0, 2.0)},
            {'name': 'O-H (hydroxyl / defect)', 'be': 531.5, 'be_tol': 0.4, 'fwhm': (1.2, 2.2)},
            {'name': 'Adsorbed water / surface contamination', 'be': 532.5, 'be_tol': 0.5, 'fwhm': (1.3, 2.5)},
        ],
        'optional_components': [
            {'name': 'S=O (sulfonate group)', 'be': 531.6, 'be_tol': 0.3, 'fwhm': (1.0, 1.8),
             'hint': '예: sulfate, sulfonate, TFSI/TFS 이온성 액체'},
            {'name': 'O-C=O (carboxylate organic)', 'be': 533.5, 'be_tol': 0.3, 'fwhm': (1.0, 2.0),
             'hint': '예: 표면 흡착 organic acid'},
        ],
    },
    'Polymer with O (O1s)': {
        'region': 'O1s',
        'description': 'Organic polymer with oxygen functionalities (PMMA, PEG, PVA 등)',
        'reference': 'Beamson & Briggs High-Res XPS of Organic Polymers',
        'components': [
            {'name': 'C=O (carbonyl)',  'be': 531.3, 'be_tol': 0.3, 'fwhm': (1.2, 2.0)},
            {'name': 'C-O (ether / alcohol)', 'be': 532.8, 'be_tol': 0.3, 'fwhm': (1.2, 2.0)},
            {'name': 'O-C=O (carboxylate / ester)', 'be': 533.5, 'be_tol': 0.3, 'fwhm': (1.2, 2.0)},
        ],
        'optional_components': [
            {'name': 'S=O (sulfonate group)', 'be': 531.6, 'be_tol': 0.3, 'fwhm': (1.0, 1.8),
             'hint': '예: 황화 polymer (Nafion 등), sulfonated 그룹'},
            {'name': 'M-O (metal oxide impurity)', 'be': 530.0, 'be_tol': 0.5, 'fwhm': (1.0, 2.0),
             'hint': '예: 표면 metal oxide 잔류물'},
        ],
    },

    # ============================================================
    # C1s region (280-295 eV)
    # ============================================================
    'Polymer (C1s)': {
        'region': 'C1s',
        'description': 'Typical organic polymer C1s (PE, PMMA, PS 등)',
        'reference': 'Beamson & Briggs',
        'components': [
            {'name': 'C-C / C-H (aliphatic)', 'be': 284.8, 'be_tol': 0.1, 'fwhm': (0.9, 1.5)},
            {'name': 'C-O (ether / alcohol)', 'be': 286.3, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
            {'name': 'C=O (carbonyl)',        'be': 287.8, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
            {'name': 'O-C=O (carboxylate / ester)', 'be': 288.9, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
        ],
        'optional_components': [
            {'name': 'C-F (covalent fluoride)', 'be': 290.5, 'be_tol': 0.5, 'fwhm': (1.0, 1.6),
             'hint': '예: 부분 fluorinated polymer (PVDF 등)'},
            {'name': 'CF2 (perfluoro)', 'be': 291.5, 'be_tol': 0.3, 'fwhm': (1.0, 1.6),
             'hint': '예: PTFE, Nafion 같은 perfluoro polymer'},
            {'name': 'CF3 (terminal trifluoro)', 'be': 293.5, 'be_tol': 0.3, 'fwhm': (1.0, 1.6),
             'hint': '예: TFSI 이온성 액체, trifluoromethyl 그룹'},
            {'name': 'C-N', 'be': 286.0, 'be_tol': 0.3, 'fwhm': (1.0, 1.6),
             'hint': '예: amine, amide 함유 polymer'},
            {'name': 'π-π* shake-up', 'be': 290.5, 'be_tol': 0.5, 'fwhm': (1.5, 3.0),
             'hint': '예: aromatic polymer (PS, PEEK 등)'},
        ],
    },
    'Graphitic carbon (C1s)': {
        'region': 'C1s',
        'description': 'Graphite, graphene, CNT, carbon nanofiber',
        'reference': 'Carbon 65, 249 (2013)',
        'components': [
            {'name': 'sp2 C=C (graphitic)',  'be': 284.4, 'be_tol': 0.1, 'fwhm': (0.7, 1.2)},
            {'name': 'sp3 C-C (defect)',     'be': 285.2, 'be_tol': 0.2, 'fwhm': (0.9, 1.5)},
            {'name': 'C-O (oxidized surface)', 'be': 286.3, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
            {'name': 'C=O (carbonyl on edge)', 'be': 287.8, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
            {'name': 'π-π* shake-up',          'be': 290.5, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
        ],
        'optional_components': [
            {'name': 'O-C=O (carboxylate edge)', 'be': 288.9, 'be_tol': 0.3, 'fwhm': (1.0, 1.6),
             'hint': '예: 산화된 graphene edge group'},
            {'name': 'C-F', 'be': 290.5, 'be_tol': 0.5, 'fwhm': (1.0, 1.6),
             'hint': '예: fluorinated graphene'},
        ],
    },

    # ============================================================
    # F1s region (680-695 eV)
    # ============================================================
    'Fluorinated (F1s)': {
        'region': 'F1s',
        'description': '불소 함유 화합물 일반 (metal fluoride, fluorinated polymer 등)',
        'reference': 'NIST XPS DB',
        'components': [
            {'name': 'M-F (ionic / metal fluoride)', 'be': 685.5, 'be_tol': 0.5, 'fwhm': (1.2, 2.5)},
            {'name': 'C-F (covalent / organic)',     'be': 688.5, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
        ],
        'optional_components': [
            {'name': 'F (semi-ionic, e.g. graphite-F)', 'be': 687.0, 'be_tol': 0.5, 'fwhm': (1.5, 2.5),
             'hint': '예: fluorinated graphene, GIC 등 중간 결합'},
        ],
    },

    # ============================================================
    # N1s region (395-410 eV)
    # ============================================================
    'Nitrogen-containing (N1s)': {
        'region': 'N1s',
        'description': 'Polymer/MOF/biomolecule 등 질소 함유 일반',
        'reference': 'Beamson & Briggs; NIST',
        'components': [
            {'name': 'C-N (amine / amide)',  'be': 399.5, 'be_tol': 0.5, 'fwhm': (1.0, 2.0)},
            {'name': 'C=N (imine / pyridinic)', 'be': 398.5, 'be_tol': 0.5, 'fwhm': (1.0, 2.0)},
        ],
        'optional_components': [
            {'name': 'N-O (nitro / nitrite)', 'be': 405.0, 'be_tol': 0.8, 'fwhm': (1.2, 2.5),
             'hint': '예: nitro group (-NO2), nitrite'},
            {'name': 'N+ (protonated / ammonium)', 'be': 401.5, 'be_tol': 0.5, 'fwhm': (1.0, 2.0),
             'hint': '예: 양이온성 N (-NH3+, imidazolium 등)'},
            {'name': 'Pyrrolic N', 'be': 400.5, 'be_tol': 0.4, 'fwhm': (1.2, 2.0),
             'hint': '예: 질소 도핑 carbon (N-doped graphene)'},
            {'name': 'Graphitic N', 'be': 401.0, 'be_tol': 0.4, 'fwhm': (1.2, 2.0),
             'hint': '예: 질소 도핑 carbon (graphitic substitution)'},
        ],
    },
}


# ===================================================================
# Hierarchical Material Library (v1.2)
# ===================================================================
# 사용자가 "재료군 → 구체 재료 → 컴포넌트" 단계로 선택하는 새 구조.
# 기존 MATERIAL_TEMPLATES도 유지 (multi-match 호환성).
#
# Region별 BE/FWHM은 NIST XPS DB + 분야별 표준 논문 기반.
# OV (Oxygen Vacancy) 위치는 재료별로 다름 — 논문 그대로 유지.
# ===================================================================

MATERIAL_HIERARCHY = {
    # ============================================================
    # Metal Oxide 군 — O1s region
    # ============================================================
    'Metal Oxide': {
        'region': 'O1s',
        'description': '금속 산화물 (TiO2, SnO2, ZnO 등)',
        'materials': {
            'TiO2': {
                'description': 'Titanium dioxide — 광촉매, 페로브스카이트 ETL',
                'reference': 'Nat. Commun. 15, 9435 (2024); NIST XPS DB',
                'components': [
                    {'name': 'TiO2 lattice O', 'be': 530.5, 'be_tol': 0.3, 'fwhm': (1.0, 2.0)},
                    {'name': 'OV (Oxygen Vacancy)', 'be': 531.3, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                    {'name': 'Non-lattice O / OH', 'be': 532.1, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                ],
                'optional_components': [
                    {'name': 'Adsorbed water', 'be': 533.0, 'be_tol': 0.4, 'fwhm': (1.2, 2.5),
                     'hint': '예: 표면 흡착 H2O, ambient 노출 시 빈번'},
                ],
            },
            'SnO2': {
                'description': 'Tin dioxide — TCO, 페로브스카이트 ETL',
                'reference': 'NIST; Wang et al. JMCA 2017',
                'components': [
                    {'name': 'SnO2 lattice O', 'be': 530.6, 'be_tol': 0.3, 'fwhm': (1.0, 2.0)},
                    {'name': 'OV (Oxygen Vacancy)', 'be': 531.5, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                    {'name': 'OH / adsorbed', 'be': 532.4, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                ],
            },
            'ZnO': {
                'description': 'Zinc oxide — TCO, 광촉매, ETL',
                'reference': 'NIST XPS DB',
                'components': [
                    {'name': 'ZnO lattice O', 'be': 530.2, 'be_tol': 0.3, 'fwhm': (1.0, 2.0)},
                    {'name': 'OV (Oxygen Vacancy)', 'be': 531.2, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                    {'name': 'OH / adsorbed water', 'be': 532.4, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                ],
            },
            'Fe2O3': {
                'description': 'Hematite (α-Fe2O3) — 광촉매, 자성',
                'reference': 'NIST; Yamashita & Hayes (2008)',
                'components': [
                    {'name': 'Fe-O lattice', 'be': 530.0, 'be_tol': 0.3, 'fwhm': (1.0, 2.0)},
                    {'name': 'OH / FeOOH', 'be': 531.5, 'be_tol': 0.4, 'fwhm': (1.2, 2.2)},
                ],
                'optional_components': [
                    {'name': 'OV (Oxygen Vacancy)', 'be': 531.0, 'be_tol': 0.4, 'fwhm': (1.0, 2.0),
                     'hint': '환원된 hematite 또는 nano structure에서 관찰'},
                    {'name': 'Adsorbed water', 'be': 533.0, 'be_tol': 0.4, 'fwhm': (1.2, 2.5),
                     'hint': '예: ambient 노출 시'},
                ],
            },
            'NiO': {
                'description': 'Nickel oxide — pin-type 페로브스카이트 HTL',
                'reference': 'NIST; Biesinger et al. Surf. Interface Anal. (2011)',
                'components': [
                    {'name': 'NiO lattice O', 'be': 529.6, 'be_tol': 0.3, 'fwhm': (0.9, 1.8)},
                    {'name': 'Ni3+ surface (Ni2O3)', 'be': 531.1, 'be_tol': 0.3, 'fwhm': (1.0, 2.0)},
                    {'name': 'OH / adsorbed', 'be': 532.5, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                ],
            },
            'Cu2O / CuO': {
                'description': 'Copper oxides — Cu2O와 CuO는 BE 약간 다름',
                'reference': 'NIST; Biesinger Appl. Surf. Sci. (2017)',
                'components': [
                    {'name': 'Cu-O lattice', 'be': 530.2, 'be_tol': 0.5, 'fwhm': (1.0, 2.0)},
                    {'name': 'OH / Cu(OH)2', 'be': 531.3, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                ],
                'optional_components': [
                    {'name': 'Adsorbed water', 'be': 533.0, 'be_tol': 0.4, 'fwhm': (1.2, 2.5),
                     'hint': '예: ambient 노출 시'},
                ],
            },
            'Al2O3': {
                'description': 'Aluminum oxide — 절연체, 패시베이션 layer',
                'reference': 'NIST; Rotole & Sherwood (1998)',
                'components': [
                    {'name': 'Al-O lattice', 'be': 531.0, 'be_tol': 0.3, 'fwhm': (1.2, 2.0)},
                    {'name': 'OH / hydroxide', 'be': 532.5, 'be_tol': 0.3, 'fwhm': (1.2, 2.2)},
                ],
                'optional_components': [
                    {'name': 'Adsorbed water', 'be': 533.5, 'be_tol': 0.4, 'fwhm': (1.2, 2.5),
                     'hint': '예: 표면 흡착 H2O'},
                ],
            },
            'In2O3': {
                'description': 'Indium oxide — ITO 기판의 In 성분',
                'reference': 'NIST; Donley et al. Langmuir (2002)',
                'components': [
                    {'name': 'In-O lattice', 'be': 530.0, 'be_tol': 0.3, 'fwhm': (1.0, 2.0)},
                    {'name': 'OH / defect', 'be': 531.5, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                ],
                'optional_components': [
                    {'name': 'Adsorbed water', 'be': 532.5, 'be_tol': 0.4, 'fwhm': (1.2, 2.5),
                     'hint': '예: ITO 표면 흡착'},
                ],
            },
            'Generic (other oxides)': {
                'description': '위에 없는 일반 단일 금속 산화물',
                'reference': 'NIST XPS DB (typical values)',
                'components': [
                    {'name': 'M-O (lattice oxygen)', 'be': 530.0, 'be_tol': 0.5, 'fwhm': (1.0, 2.0)},
                    {'name': 'O-H / defect', 'be': 531.5, 'be_tol': 0.4, 'fwhm': (1.2, 2.2)},
                    {'name': 'Adsorbed water / contamination', 'be': 532.5, 'be_tol': 0.5, 'fwhm': (1.3, 2.5)},
                ],
                'optional_components': [
                    {'name': 'OV (Oxygen Vacancy)', 'be': 531.2, 'be_tol': 0.5, 'fwhm': (1.0, 2.2),
                     'hint': '재료별로 위치 다름 — 일반적 추정값'},
                    {'name': 'C=O (carbonate / surface contamination)', 'be': 533.0, 'be_tol': 0.5, 'fwhm': (1.0, 2.0),
                     'hint': '예: 표면 carbonate (CO₃²⁻), 유기 오염물 등 — 무기 샘플엔 보통 무관'},
                ],
            },
            'Mixed oxide (binary, e.g. Al2O3-SnO2, ITO)': {
                'description': ('두 금속 산화물이 혼재된 박막 또는 분말 — '
                                'Al₂O₃-SnO₂, ITO (In₂O₃-SnO₂), TiO₂-SnO₂, '
                                'core-shell 산화물 등'),
                'reference': 'NIST + 분야별 표준값',
                'components': [
                    # 두 metal-O lattice 환경 — 위치는 사용자가 자기 재료에 맞게 조정 권장
                    {'name': 'M1-O lattice (lower BE metal)', 'be': 530.0, 'be_tol': 0.7, 'fwhm': (1.0, 2.0)},
                    {'name': 'M2-O lattice (higher BE metal)', 'be': 531.0, 'be_tol': 0.7, 'fwhm': (1.0, 2.0)},
                    {'name': 'O-H / defect', 'be': 531.8, 'be_tol': 0.4, 'fwhm': (1.2, 2.2)},
                    {'name': 'Adsorbed water / contamination', 'be': 532.8, 'be_tol': 0.5, 'fwhm': (1.3, 2.5)},
                ],
                'optional_components': [
                    {'name': 'OV (Oxygen Vacancy)', 'be': 531.3, 'be_tol': 0.5, 'fwhm': (1.0, 2.2),
                     'hint': '결함 산화물 또는 환원 표면에서 관찰'},
                ],
                'note': ('두 metal-O lattice의 BE는 각 재료마다 다릅니다. '
                          '예: Al₂O₃ ~531, SnO₂ ~530.6, In₂O₃ ~530.0. '
                          '본인 재료에 맞춰 BE 중심을 직접 조정하세요. '
                          'M1=낮은 BE, M2=높은 BE로 자동 정렬됩니다.'),
            },
        },
    },

    # ============================================================
    # MOF 군 — O1s region
    # ============================================================
    'MOF': {
        'region': 'O1s',
        'description': '금속-유기 골격체 (Metal-Organic Framework)',
        'materials': {
            'Zr-based': {
                'description': 'Zr-oxo cluster + carboxylate linker',
                'reference': 'Nat. Commun. 16, 162 (2025); Wang et al. JMCA 2017',
                'components': [
                    {'name': 'Zr-O-Zr (oxo cluster)', 'be': 529.4, 'be_tol': 0.3, 'fwhm': (0.9, 2.0)},
                    {'name': 'Zr-O-C (carboxylate linker)', 'be': 530.7, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                    {'name': 'Zr-O-H (hydroxyl / adsorbed water)', 'be': 532.1, 'be_tol': 0.2, 'fwhm': (1.0, 2.2)},
                ],
                'optional_components': [
                    {'name': 'S=O (sulfonate group)', 'be': 531.6, 'be_tol': 0.3, 'fwhm': (1.0, 1.8),
                     'hint': '예: sulfate, sulfonate, TFSI/TFS 이온성 액체, 황 함유 분자'},
                    {'name': 'C=O (carbonyl)', 'be': 531.3, 'be_tol': 0.3, 'fwhm': (1.0, 2.0),
                     'hint': '예: carbonyl, ketone, aldehyde'},
                    {'name': 'O-C=O (carboxylate organic)', 'be': 533.5, 'be_tol': 0.3, 'fwhm': (1.0, 2.0),
                     'hint': '예: 자유 carboxylic acid (linker가 아닌)'},
                ],
            },
            'Generic (other metal MOFs)': {
                'description': '다른 metal MOF (Cu, Fe, Zn 기반 등)',
                'reference': 'NIST + 분야별 표준값 (재료별 미세 차이 있음)',
                'components': [
                    {'name': 'M-O-M (metal-oxo)', 'be': 529.8, 'be_tol': 0.5, 'fwhm': (0.9, 2.0)},
                    {'name': 'M-O-C (carboxylate linker)', 'be': 531.0, 'be_tol': 0.4, 'fwhm': (1.0, 2.2)},
                    {'name': 'M-O-H / hydroxyl', 'be': 532.2, 'be_tol': 0.3, 'fwhm': (1.0, 2.2)},
                ],
                'optional_components': [
                    {'name': 'S=O (sulfonate group)', 'be': 531.6, 'be_tol': 0.3, 'fwhm': (1.0, 1.8),
                     'hint': '예: TFSI 등 황 함유'},
                    {'name': 'C=O (carbonyl)', 'be': 531.3, 'be_tol': 0.3, 'fwhm': (1.0, 2.0),
                     'hint': '예: 자유 carbonyl'},
                ],
            },
        },
    },

    # ============================================================
    # Polymer 군
    # ============================================================
    'Polymer': {
        'region': 'O1s',  # 기본은 O1s, 하위에서 region 다를 수 있음
        'description': '유기 고분자',
        'materials': {
            'With O functionalities (O1s)': {
                'description': 'PMMA, PEG, PVA 등 산소 함유 polymer',
                'reference': 'Beamson & Briggs High-Res XPS of Organic Polymers',
                'region': 'O1s',
                'components': [
                    {'name': 'C=O (carbonyl)', 'be': 531.3, 'be_tol': 0.3, 'fwhm': (1.2, 2.0)},
                    {'name': 'C-O (ether / alcohol)', 'be': 532.8, 'be_tol': 0.3, 'fwhm': (1.2, 2.0)},
                    {'name': 'O-C=O (carboxylate / ester)', 'be': 533.5, 'be_tol': 0.3, 'fwhm': (1.2, 2.0)},
                ],
                'optional_components': [
                    {'name': 'S=O (sulfonate group)', 'be': 531.6, 'be_tol': 0.3, 'fwhm': (1.0, 1.8),
                     'hint': '예: 황화 polymer (Nafion 등)'},
                    {'name': 'M-O (metal oxide impurity)', 'be': 530.0, 'be_tol': 0.5, 'fwhm': (1.0, 2.0),
                     'hint': '예: 표면 metal oxide 잔류물'},
                ],
            },
            'C1s — typical organic polymer': {
                'description': 'PE, PMMA, PS 등 일반 polymer C1s',
                'reference': 'Beamson & Briggs',
                'region': 'C1s',
                'components': [
                    {'name': 'C-C / C-H (aliphatic)', 'be': 284.8, 'be_tol': 0.1, 'fwhm': (0.9, 1.5)},
                    {'name': 'C-O (ether / alcohol)', 'be': 286.3, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
                    {'name': 'C=O (carbonyl)', 'be': 287.8, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
                    {'name': 'O-C=O (carboxylate / ester)', 'be': 288.9, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
                ],
                'optional_components': [
                    {'name': 'C-F (covalent fluoride)', 'be': 290.5, 'be_tol': 0.5, 'fwhm': (1.0, 1.6),
                     'hint': '예: PVDF 등'},
                    {'name': 'CF2 (perfluoro)', 'be': 291.5, 'be_tol': 0.3, 'fwhm': (1.0, 1.6),
                     'hint': '예: PTFE, Nafion'},
                    {'name': 'CF3 (terminal trifluoro)', 'be': 293.5, 'be_tol': 0.3, 'fwhm': (1.0, 1.6),
                     'hint': '예: TFSI 이온성 액체'},
                    {'name': 'C-N', 'be': 286.0, 'be_tol': 0.3, 'fwhm': (1.0, 1.6),
                     'hint': '예: amine, amide 함유'},
                    {'name': 'π-π* shake-up', 'be': 290.5, 'be_tol': 0.5, 'fwhm': (1.5, 3.0),
                     'hint': '예: aromatic polymer'},
                ],
            },
            'Graphitic carbon (C1s)': {
                'description': '흑연/graphene/CNT/CNF',
                'reference': 'Carbon 65, 249 (2013)',
                'region': 'C1s',
                'components': [
                    {'name': 'sp2 C=C (graphitic)', 'be': 284.4, 'be_tol': 0.1, 'fwhm': (0.7, 1.2)},
                    {'name': 'sp3 C-C (defect)', 'be': 285.2, 'be_tol': 0.2, 'fwhm': (0.9, 1.5)},
                    {'name': 'C-O (oxidized surface)', 'be': 286.3, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
                    {'name': 'C=O (carbonyl on edge)', 'be': 287.8, 'be_tol': 0.3, 'fwhm': (1.0, 1.6)},
                    {'name': 'π-π* shake-up', 'be': 290.5, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                ],
                'optional_components': [
                    {'name': 'O-C=O (carboxylate edge)', 'be': 288.9, 'be_tol': 0.3, 'fwhm': (1.0, 1.6),
                     'hint': '예: 산화된 graphene edge'},
                    {'name': 'C-F', 'be': 290.5, 'be_tol': 0.5, 'fwhm': (1.0, 1.6),
                     'hint': '예: fluorinated graphene'},
                ],
            },
        },
    },

    # ============================================================
    # Multi-oxidation states 군 — 한 원소의 여러 산화수 공존
    # ============================================================
    'Multi-oxidation states': {
        'region': 'mixed',
        'description': '같은 원소의 여러 산화수가 공존 (페로브스카이트, 배터리, 촉매 등)',
        'materials': {
            'Tin oxidation states (Sn3d)': {
                'description': ('Sn²⁺/Sn⁴⁺ 공존 — perovskite solar cells, '
                                'Li/Na-ion battery anodes, 표면 산화된 Sn 박막, '
                                'SnO₂ 광촉매 등'),
                'reference': 'Nat. Commun. 15, 9435 (2024); NIST XPS DB',
                'region': 'Sn3d',
                'components': [
                    # Sn²⁺ doublet — main + minor (spin-orbit 8.41 eV)
                    # be_tol=1.0: 충전(charging) 시프트 흡수용 (절연체 페로브스카이트는 ±1 eV 흔함)
                    # fwhm 0.7~1.2: 좁은 피크에 맞춤 (NIST + 논문 표준)
                    {'name': 'Sn²⁺ 3d₅/₂', 'be': 486.6, 'be_tol': 1.0, 'fwhm': (0.7, 1.2)},
                    {'name': 'Sn²⁺ 3d₃/₂', 'be': 495.0, 'be_tol': 1.0, 'fwhm': (0.7, 1.2)},
                    # Sn⁴⁺ doublet — main + minor (보통 약간 더 넓음)
                    {'name': 'Sn⁴⁺ 3d₅/₂', 'be': 487.5, 'be_tol': 1.0, 'fwhm': (0.8, 1.5)},
                    {'name': 'Sn⁴⁺ 3d₃/₂', 'be': 495.9, 'be_tol': 1.0, 'fwhm': (0.8, 1.5)},
                ],
                'note': ('Sn²⁺와 Sn⁴⁺의 BE 차이는 ~0.9 eV로 작음. '
                         'be_tol=1.0은 충전(charging) 시프트 흡수용. '
                         'FWHM 공유 + η 공유 권장.'),
            },
            'Copper oxidation states (Cu2p)': {
                'description': ('Cu⁺/Cu²⁺ 공존 — heterogeneous catalysts, '
                                'electrocatalysis (CO₂RR/HER/OER), Cu corrosion, '
                                'copper-based MOFs, photocatalysts 등'),
                'reference': 'Biesinger Appl. Surf. Sci. (2017); NIST',
                'region': 'Cu2p',
                'components': [
                    {'name': 'Cu⁺ 2p₃/₂', 'be': 932.4, 'be_tol': 0.4, 'fwhm': (0.9, 2.0)},
                    {'name': 'Cu⁺ 2p₁/₂', 'be': 952.2, 'be_tol': 0.4, 'fwhm': (0.9, 2.0)},
                    {'name': 'Cu²⁺ 2p₃/₂', 'be': 933.7, 'be_tol': 0.4, 'fwhm': (1.0, 2.5)},
                    {'name': 'Cu²⁺ 2p₁/₂', 'be': 953.5, 'be_tol': 0.4, 'fwhm': (1.0, 2.5)},
                ],
                'optional_components': [
                    {'name': 'Cu²⁺ satellite (~940 eV)', 'be': 940.0, 'be_tol': 1.0, 'fwhm': (2.0, 5.0),
                     'hint': 'Cu²⁺의 강한 shake-up satellite, 정량 시 중요'},
                    {'name': 'Cu²⁺ satellite (~960 eV)', 'be': 960.0, 'be_tol': 1.0, 'fwhm': (2.0, 5.0),
                     'hint': 'Cu²⁺의 2p1/2 satellite'},
                ],
                'note': 'Cu²⁺는 satellite peak이 강해서 옵션 추가를 권장합니다.',
            },
            'Iron oxidation states (Fe2p)': {
                'description': ('Fe²⁺/Fe³⁺ 공존 — Fe₃O₄ (magnetite), '
                                'partially-oxidized FeO, Fe-MOFs, biological samples, '
                                'corrosion 표면 등'),
                'reference': 'Yamashita & Hayes (2008); Biesinger',
                'region': 'Fe2p',
                'components': [
                    {'name': 'Fe²⁺ 2p₃/₂', 'be': 709.8, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                    {'name': 'Fe²⁺ 2p₁/₂', 'be': 723.0, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                    {'name': 'Fe³⁺ 2p₃/₂', 'be': 711.0, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                    {'name': 'Fe³⁺ 2p₁/₂', 'be': 724.5, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                ],
                'optional_components': [
                    {'name': 'Fe²⁺ satellite (~715 eV)', 'be': 715.0, 'be_tol': 1.0, 'fwhm': (2.0, 4.0),
                     'hint': 'Fe²⁺ shake-up'},
                    {'name': 'Fe³⁺ satellite (~719 eV)', 'be': 719.0, 'be_tol': 1.0, 'fwhm': (2.0, 4.0),
                     'hint': 'Fe³⁺ shake-up'},
                ],
                'note': 'Fe도 satellite peak 강함. 분리 매우 어려운 케이스 — 결과 해석 주의.',
            },
            'Titanium oxidation states (Ti2p)': {
                'description': ('Ti³⁺/Ti⁴⁺ 공존 — defective TiO₂ (광촉매, ETL), '
                                'Ti suboxides (TiO_{2-x}), reduced TiO₂ surfaces, '
                                'Ti-based perovskites 등'),
                'reference': 'NIST; Diebold Surf. Sci. Rep. (2003)',
                'region': 'Ti2p',
                'components': [
                    {'name': 'Ti³⁺ 2p₃/₂', 'be': 457.7, 'be_tol': 0.4, 'fwhm': (1.0, 2.0)},
                    {'name': 'Ti³⁺ 2p₁/₂', 'be': 463.4, 'be_tol': 0.4, 'fwhm': (1.0, 2.0)},
                    {'name': 'Ti⁴⁺ 2p₃/₂', 'be': 458.8, 'be_tol': 0.4, 'fwhm': (0.9, 1.8)},
                    {'name': 'Ti⁴⁺ 2p₁/₂', 'be': 464.5, 'be_tol': 0.4, 'fwhm': (0.9, 1.8)},
                ],
                'note': 'Ti³⁺는 oxygen vacancy의 신호 — defective TiO₂에서 관찰',
            },
            'Manganese oxidation states (Mn2p)': {
                'description': ('Mn²⁺/Mn³⁺/Mn⁴⁺ 공존 — Li-ion battery cathodes '
                                '(LMO, NMC, LiMn₂O₄), water-splitting catalysts, '
                                'Mn oxide thin films (MnO/Mn₂O₃/MnO₂) 등'),
                'reference': 'Biesinger Appl. Surf. Sci. (2011); battery literature',
                'region': 'Mn2p',
                'components': [
                    {'name': 'Mn²⁺ 2p₃/₂', 'be': 640.7, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                    {'name': 'Mn²⁺ 2p₁/₂', 'be': 652.2, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                    {'name': 'Mn⁴⁺ 2p₃/₂', 'be': 642.5, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                    {'name': 'Mn⁴⁺ 2p₁/₂', 'be': 654.1, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                ],
                'optional_components': [
                    {'name': 'Mn³⁺ 2p₃/₂', 'be': 641.5, 'be_tol': 0.5, 'fwhm': (1.5, 3.0),
                     'hint': '중간 산화수 — Mn₂O₃, Mn₃O₄에서 관찰'},
                    {'name': 'Mn³⁺ 2p₁/₂', 'be': 653.0, 'be_tol': 0.5, 'fwhm': (1.5, 3.0),
                     'hint': 'Mn³⁺의 minor doublet'},
                ],
                'note': 'Mn은 3개 산화수 모두 가능 — Mn³⁺는 옵션으로 추가하세요.',
            },
        },
    },

    # ============================================================
    # 단순 region 모음 (재료 정보 없을 때 fallback)
    # ============================================================
    'Other (region only)': {
        'region': 'mixed',
        'description': '재료 정보 없이 region만 알 때',
        'materials': {
            'F1s (fluorinated)': {
                'description': '불소 함유 화합물 일반',
                'reference': 'NIST XPS DB',
                'region': 'F1s',
                'components': [
                    {'name': 'M-F (ionic / metal fluoride)', 'be': 685.5, 'be_tol': 0.5, 'fwhm': (1.2, 2.5)},
                    {'name': 'C-F (covalent / organic)', 'be': 688.5, 'be_tol': 0.5, 'fwhm': (1.5, 3.0)},
                ],
                'optional_components': [
                    {'name': 'F (semi-ionic, e.g. graphite-F)', 'be': 687.0, 'be_tol': 0.5, 'fwhm': (1.5, 2.5),
                     'hint': '예: fluorinated graphene, GIC'},
                ],
            },
            'N1s (nitrogen-containing)': {
                'description': '질소 함유 화합물 일반',
                'reference': 'Beamson & Briggs; NIST',
                'region': 'N1s',
                'components': [
                    {'name': 'C-N (amine / amide)', 'be': 399.5, 'be_tol': 0.5, 'fwhm': (1.0, 2.0)},
                    {'name': 'C=N (imine / pyridinic)', 'be': 398.5, 'be_tol': 0.5, 'fwhm': (1.0, 2.0)},
                ],
                'optional_components': [
                    {'name': 'N-O (nitro / nitrite)', 'be': 405.0, 'be_tol': 0.8, 'fwhm': (1.2, 2.5),
                     'hint': '예: -NO2, nitrite'},
                    {'name': 'N+ (protonated / ammonium)', 'be': 401.5, 'be_tol': 0.5, 'fwhm': (1.0, 2.0),
                     'hint': '예: -NH3+, imidazolium'},
                    {'name': 'Pyrrolic N', 'be': 400.5, 'be_tol': 0.4, 'fwhm': (1.2, 2.0),
                     'hint': '예: 질소 도핑 carbon'},
                    {'name': 'Graphitic N', 'be': 401.0, 'be_tol': 0.4, 'fwhm': (1.2, 2.0),
                     'hint': '예: 질소 도핑 graphene'},
                ],
            },
        },
    },
}


def get_hierarchy_families():
    """재료군 목록 반환 (1단계 dropdown용)"""
    return list(MATERIAL_HIERARCHY.keys())


def get_hierarchy_materials(family: str):
    """특정 재료군의 구체 재료 목록 (2단계 dropdown용)"""
    if family not in MATERIAL_HIERARCHY:
        return []
    return list(MATERIAL_HIERARCHY[family]['materials'].keys())


def get_hierarchy_template(family: str, material: str):
    """선택된 재료의 템플릿 dict 반환 (기존 MATERIAL_TEMPLATES와 같은 형식)"""
    if family not in MATERIAL_HIERARCHY:
        return None
    family_data = MATERIAL_HIERARCHY[family]
    if material not in family_data['materials']:
        return None
    mat = family_data['materials'][material]
    # region 결정: material에 명시되어 있으면 그것, 없으면 family region 상속
    region = mat.get('region', family_data.get('region', 'unknown'))
    return {
        'region': region,
        'description': mat['description'],
        'reference': mat.get('reference', ''),
        'components': mat['components'],
        'optional_components': mat.get('optional_components', []),
    }


def hierarchy_components(family: str, material: str,
                           include_optional: List[str] = None) -> List['ComponentSpec']:
    """계층 선택으로부터 ComponentSpec 리스트 생성 (Expert UI용)"""
    tmpl = get_hierarchy_template(family, material)
    if tmpl is None:
        raise ValueError(f"Unknown material: {family} / {material}")
    specs = []
    for c in tmpl['components']:
        specs.append(ComponentSpec(
            name=c['name'], be=c['be'], be_tol=c['be_tol'],
            fwhm_min=c['fwhm'][0], fwhm_max=c['fwhm'][1]
        ))
    include_optional = include_optional or []
    for c in tmpl.get('optional_components', []):
        if c['name'] in include_optional:
            specs.append(ComponentSpec(
                name=c['name'], be=c['be'], be_tol=c['be_tol'],
                fwhm_min=c['fwhm'][0], fwhm_max=c['fwhm'][1]
            ))
    return specs


def find_default_material_for_region(region: str):
    """
    region이 주어지면 그 region에 해당하는 (family, material) 쌍 후보 반환.
    Auto-suggest 기능에서 사용.

    Returns: list of (family, material) tuples, 가장 일반적인 것 먼저
    """
    candidates = []
    for fam_name, fam in MATERIAL_HIERARCHY.items():
        for mat_name, mat in fam['materials'].items():
            mat_region = mat.get('region', fam.get('region', ''))
            if mat_region == region or fam.get('region') == region:
                candidates.append((fam_name, mat_name))
    return candidates
@dataclass
class ComponentSpec:
    name: str
    be: float                    # 중심 BE
    be_tol: float = 0.3          # BE 이동 허용 범위 (위치 고정 시 0)
    fwhm_min: float = 0.8
    fwhm_max: float = 2.5
    lock_position: bool = False  # True면 be에 완전히 고정
    lock_fwhm: Optional[float] = None  # 값이 있으면 그 값으로 고정
    lock_eta: Optional[float] = None   # 값이 있으면 그 값으로 고정


# -------------------------------------------------------------------
# Pseudo-Voigt (import)
# -------------------------------------------------------------------
from xps_engine import pseudo_voigt, shirley_background


# -------------------------------------------------------------------
# Expert 피팅 엔진
# -------------------------------------------------------------------
def expert_fit(be, counts, components: List[ComponentSpec],
               share_fwhm: bool = False, share_eta: bool = False,
               use_shirley: bool = True, bg_kwargs=None):
    """
    사용자 정의 컴포넌트로 피팅.

    bg_kwargs: shirley_background()에 전달할 추가 옵션 dict
    """
    bg_kwargs = bg_kwargs or {}
    n = len(components)

    # 1) 배경
    if use_shirley:
        bg = shirley_background(be, counts, **bg_kwargs)
    else:
        bg = np.zeros_like(counts, dtype=float)
    y_corr = counts - bg

    # 2) 파라미터 인덱싱 설계
    # 각 컴포넌트의 자유 파라미터 개수와 위치 기록
    param_layout = []  # 각 항목: (comp_idx, param_name, p0, lo, hi)

    amax = max(y_corr)

    # 전역 shared 파라미터 준비
    global_params = []
    if share_fwhm:
        avg_fwhm = np.mean([(c.fwhm_min + c.fwhm_max) / 2 for c in components])
        fwhm_lo = max(c.fwhm_min for c in components)
        fwhm_hi = min(c.fwhm_max for c in components)
        # 안전: lo > hi 방지
        if fwhm_lo >= fwhm_hi:
            fwhm_lo, fwhm_hi = min(c.fwhm_min for c in components), max(c.fwhm_max for c in components)
        global_params.append(('fwhm_shared', avg_fwhm, fwhm_lo, fwhm_hi))
    if share_eta:
        global_params.append(('eta_shared', 0.3, 0.0, 1.0))

    # 각 컴포넌트 파라미터
    for i, c in enumerate(components):
        # amp: 항상 자유
        be_idx = int(np.argmin(np.abs(be - c.be)))
        amp0 = max(y_corr[be_idx], amax * 0.05)
        param_layout.append((i, 'amp', amp0, amp0 * 0.01, amp0 * 5.0 + 1e-6))

        # position
        if not c.lock_position:
            param_layout.append((i, 'be', c.be, c.be - c.be_tol, c.be + c.be_tol))

        # fwhm (공유가 아니고 잠김도 아닐 때만)
        if not share_fwhm and c.lock_fwhm is None:
            fwhm0 = (c.fwhm_min + c.fwhm_max) / 2
            param_layout.append((i, 'fwhm', fwhm0, c.fwhm_min, c.fwhm_max))

        # eta
        if not share_eta and c.lock_eta is None:
            param_layout.append((i, 'eta', 0.3, 0.0, 1.0))

    # 전역을 마지막에
    for gp in global_params:
        param_layout.append((-1, gp[0], gp[1], gp[2], gp[3]))

    p0 = [pl[2] for pl in param_layout]
    lo = [pl[3] for pl in param_layout]
    hi = [pl[4] for pl in param_layout]

    # 3) 모델 함수
    def model(x, *params):
        # param_layout 역해석
        values = {i: {'amp': None, 'be': components[i].be,
                     'fwhm': components[i].lock_fwhm,
                     'eta': components[i].lock_eta} for i in range(n)}
        global_fwhm, global_eta = None, None

        for k, (ci, pname, *_) in enumerate(param_layout):
            if ci == -1:
                if pname == 'fwhm_shared':
                    global_fwhm = params[k]
                elif pname == 'eta_shared':
                    global_eta = params[k]
            else:
                values[ci][pname] = params[k]

        y = np.zeros_like(x, dtype=float)
        for i, c in enumerate(components):
            amp = values[i]['amp']
            be_i = values[i]['be'] if not c.lock_position else c.be
            if share_fwhm:
                fwhm = global_fwhm
            elif c.lock_fwhm is not None:
                fwhm = c.lock_fwhm
            else:
                fwhm = values[i]['fwhm']
            if share_eta:
                eta = global_eta
            elif c.lock_eta is not None:
                eta = c.lock_eta
            else:
                eta = values[i]['eta']
            y = y + pseudo_voigt(x, amp, be_i, fwhm, eta)
        return y

    # 4) 피팅
    try:
        popt, pcov = curve_fit(model, be, y_corr, p0=p0,
                               bounds=(lo, hi), maxfev=30000)
    except Exception as e:
        return {'success': False, 'reason': f'Fit failed: {e}',
                'be': be, 'counts': counts, 'background': bg}

    # 5) 파라미터 복원
    y_fit = model(be, *popt)
    resid = y_corr - y_fit
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_corr - np.mean(y_corr)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rms = float(np.sqrt(ss_res / len(be)))

    # AIC
    N = len(be); k = len(popt)
    aic = N * np.log(ss_res / N + 1e-20) + 2 * k

    # 각 컴포넌트 값 복원
    values = {i: {'amp': None, 'be': components[i].be,
                 'fwhm': components[i].lock_fwhm,
                 'eta': components[i].lock_eta} for i in range(n)}
    global_fwhm, global_eta = None, None
    # 파라미터 오차 (대각 성분)
    perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.zeros_like(popt)
    err_map = {i: {} for i in range(n)}
    global_err = {}

    for idx, (ci, pname, *_) in enumerate(param_layout):
        if ci == -1:
            if pname == 'fwhm_shared':
                global_fwhm = popt[idx]; global_err['fwhm'] = perr[idx]
            elif pname == 'eta_shared':
                global_eta = popt[idx]; global_err['eta'] = perr[idx]
        else:
            values[ci][pname] = popt[idx]
            err_map[ci][pname] = perr[idx]

    # 6) 결과 정리
    result_components = []
    for i, c in enumerate(components):
        amp = values[i]['amp']
        be_i = values[i]['be'] if not c.lock_position else c.be
        if share_fwhm:
            fwhm = global_fwhm
        elif c.lock_fwhm is not None:
            fwhm = c.lock_fwhm
        else:
            fwhm = values[i]['fwhm']
        if share_eta:
            eta = global_eta
        elif c.lock_eta is not None:
            eta = c.lock_eta
        else:
            eta = values[i]['eta']

        comp_y = pseudo_voigt(be, amp, be_i, fwhm, eta)
        area = float(abs(np.trapezoid(comp_y, be)))
        result_components.append({
            'name': c.name,
            'amplitude': float(amp),
            'position': float(be_i),
            'fwhm': float(fwhm),
            'eta': float(eta),
            'area': area,
            'curve': comp_y,
            # 에러바
            'amp_err': float(err_map[i].get('amp', 0)),
            'be_err': float(err_map[i].get('be', 0)) if not c.lock_position else 0,
            'fwhm_err': float(err_map[i].get('fwhm', global_err.get('fwhm', 0))
                              if not share_fwhm else global_err.get('fwhm', 0)),
            # 플래그 기록
            'position_locked': c.lock_position,
            'fwhm_shared': share_fwhm,
            'eta_shared': share_eta,
        })

    # 정렬 (BE 내림차순)
    result_components.sort(key=lambda c: -c['position'])
    total_area = sum(c['area'] for c in result_components) or 1
    for c in result_components:
        c['area_pct'] = 100 * c['area'] / total_area

    return {
        'success': True,
        'mode': 'expert',
        'be': be, 'counts': counts, 'background': bg,
        'y_corrected': y_corr, 'y_fit': y_fit,
        'components': result_components,
        'r_squared': r2, 'rms': rms, 'aic': aic,
        'n_components': n,
        'n_free_params': k,
        'share_fwhm': share_fwhm, 'share_eta': share_eta,
        'shared_fwhm_value': global_fwhm if share_fwhm else None,
        'shared_eta_value': global_eta if share_eta else None,
    }


def components_from_template(template_key: str,
                              include_optional: List[str] = None) -> List[ComponentSpec]:
    """템플릿 이름으로부터 ComponentSpec 리스트 생성"""
    if template_key not in MATERIAL_TEMPLATES:
        raise ValueError(f"Unknown template: {template_key}")
    tmpl = MATERIAL_TEMPLATES[template_key]
    specs = []
    for c in tmpl['components']:
        specs.append(ComponentSpec(
            name=c['name'], be=c['be'], be_tol=c['be_tol'],
            fwhm_min=c['fwhm'][0], fwhm_max=c['fwhm'][1]
        ))
    # 옵션 컴포넌트
    include_optional = include_optional or []
    for c in tmpl.get('optional_components', []):
        if c['name'] in include_optional:
            specs.append(ComponentSpec(
                name=c['name'], be=c['be'], be_tol=c['be_tol'],
                fwhm_min=c['fwhm'][0], fwhm_max=c['fwhm'][1]
            ))
    return specs
