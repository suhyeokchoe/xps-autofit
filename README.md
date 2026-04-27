# 📊 XPS AutoFit

> 자동화된 XPS 피크 피팅 웹 도구 · 공익 목적의 학술 도구

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://xps-autofit-suhyeok.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org)

XPS(X-ray Photoelectron Spectroscopy) 데이터를 업로드하면 자동으로
배경 제거 · 피크 감지 · 라인쉐입 피팅을 수행합니다. CasaXPS와 같은
상용 도구의 대안으로 학부생 · 대학원생 · 연구자가 무료로 사용할 수 있는
공익 목적의 도구를 지향합니다.

---

## 🚀 Live Demo

**👉 [https://xps-autofit-suhyeok.streamlit.app](https://xps-autofit-suhyeok.streamlit.app)**

브라우저에서 즉시 사용 가능. 설치 불필요.

---

## ✨ 주요 기능

### 🤖 자동 모드
- **Shirley 배경 보정** (iterative)
- **자동 피크 감지** (1차 + 2차 미분 기반 shoulder 검출)
- **AIC 기반 모델 선택**으로 과적합 방지
- **스핀-오빗 doublet 자동 처리** (Sn 3d, Cu 2p 등 15종 원소)

### 🔬 Expert 모드
- **재료 템플릿 라이브러리**: MOF, Metal oxide, Polymer, Graphite 등
- **제약 옵션**: 위치 고정, FWHM 공유, η 공유
- **자유도 5~12 조절** 가능 → 논문 수준 피팅
- **정직성 체크**: 데이터가 지지하지 않는 컴포넌트는 면적이 작게 표시

### 📊 결과 export
- 파라미터 CSV
- 피팅 곡선 CSV
- 고해상도 PNG

---

## 🧪 지원 Region

| Region | Singlet | Doublet |
|---|---|---|
| s-orbital | F1s, C1s, O1s, N1s | — |
| 2p (ratio 2:1) | — | Cu, Ti, Si, Fe, Ni |
| 3d (ratio 3:2) | — | Sn, In, Mo, Ag |
| 4f (ratio 4:3) | — | Au, W |

---

## 📖 사용법

1. 사이드바에서 **CSV 파일 업로드** (CasaXPS export 또는 2열 BE/Counts CSV)
2. **자동 모드** 또는 **Expert 모드** 선택
3. (Expert 모드) 재료 템플릿 선택 → 컴포넌트 편집 → 제약 옵션 설정
4. 피팅 실행 → 결과 확인 → CSV/PNG 다운로드

> ⚠️ **`.xls` 파일은 Excel에서 `다른 이름으로 저장` → CSV로 변환 후 업로드하세요.**

---

## 🛠️ 기술 스택

- **Python** 3.11+
- **Streamlit** — 웹 UI
- **NumPy / SciPy** — 수치 연산 및 비선형 최소제곱
- **Matplotlib / Pandas** — 시각화 및 데이터 처리

핵심 알고리즘:
- Shirley iterative background
- Pseudo-Voigt (= CasaXPS의 SGL) 라인쉐입
- Levenberg-Marquardt 비선형 최소제곱
- Akaike Information Criterion (AIC) 모델 선택

---

## ⚖️ 이용 정책

이 도구는 **공익 목적의 학술 도구**입니다.
- ✅ 연구·교육·논문 작성에 자유롭게 사용 가능
- ✅ 출판물에 결과 사용 시 별도 허가 불필요 (인용은 환영)
- ⚠️ 자동 피팅 결과는 **반드시 도메인 지식으로 검증**하세요
- ⚠️ R²가 높다고 올바른 모델이 아닙니다 (overfitting 주의)

업로드된 데이터는 서버 메모리에서만 처리되며, 세션 종료 시 자동 삭제됩니다.

---

## 🤝 Authors & Contributors

상세 내용은 [AUTHORS.md](./AUTHORS.md) 참조.

---

## 📚 References

- CasaXPS Cookbook (Casa Software Ltd, 2019)
- Shirley, D. A. (1972). High-resolution X-ray photoemission spectrum of the valence bands of gold. *Phys. Rev. B*, 5(12), 4709.
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Trans. Auto. Control*, 19(6), 716.
- NIST X-ray Photoelectron Spectroscopy Database

---

## 📝 License

MIT License — see [LICENSE](./LICENSE) for details.

---

*Built with ❤️ for the materials science community.*
