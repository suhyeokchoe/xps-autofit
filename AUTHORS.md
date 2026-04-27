# Authors & Contributors

## Lead Developer

**Suhyeok Choe** ([@suhyeokchoe](https://github.com/suhyeokchoe))

- B.S. Candidate, Materials Science and Engineering
- 경북대학교 신소재공학과 (Kyungpook National University, South Korea)
- Research interests: Surface analysis (XPS), materials characterization,
  data-driven research tools, semiconductor materials
- Contact: heyok7714@knu.ac.kr

---

## Project Background

XPS AutoFit은 학부 연구 과정에서 마주친 **수동 XPS 피팅의 비효율성**을
해결하고자 시작되었습니다. CasaXPS 같은 기존 도구는 강력하지만,
화학 환경별 컴포넌트 위치를 사용자가 일일이 입력해야 하는 노가다가
연구자의 시간을 크게 소모합니다.

이 프로젝트는 다음 가설을 검증합니다:

> **"전문가의 판단 규칙(재료별 표준 컴포넌트, 물리 제약)을
> 코드에 내장하면, AI/ML 없이도 논문 수준의 자동 피팅이 가능하다."**

2026년 4월 개발 시작. 공익 목적의 학술 도구로 무상 제공됩니다.

---

## Acknowledgments

- **경북대학교 신소재공학부** XPS 측정 데이터 제공
- **CasaXPS Cookbook** (Casa Software Ltd, 2019) — 수치해석 알고리즘 참고
- **NIST X-ray Photoelectron Spectroscopy Database** — 화학상태 reference 자료
- **Streamlit Community Cloud** — 무료 호스팅 인프라 제공
- **scipy / numpy** 오픈소스 커뮤니티 — 비선형 최소제곱·신호처리 라이브러리

논문 데이터 검증에 사용된 reference:
- Kim et al. (2025). *Modulating synaptic plasticity with metal-organic
  framework for information-filterable artificial retina.*
  *Nature Communications*, 16, 162.

---

## How to Contribute

이 프로젝트는 학술 커뮤니티의 기여를 환영합니다:

- **버그 신고**: [GitHub Issues](https://github.com/suhyeokchoe/xps-autofit/issues)
- **기능 제안**: 위와 동일하게 Issue로
- **새 재료 템플릿 제안**: 본인 분야에서 자주 쓰는 재료의 표준 컴포넌트 정보를
  Issue로 알려주시면 라이브러리에 추가합니다
- **데이터 기여**: 검증용 reference 데이터 (재료 + 측정 조건 명시) 제공 가능

연락: heyok7714@knu.ac.kr

---

## Citation (선택)

본 도구를 연구에 활용하셨다면 다음과 같이 인용해주시면 감사하겠습니다:

```
Choe, S. (2026). XPS AutoFit: Automated XPS peak fitting tool with
material-aware constraint fitting. https://github.com/suhyeokchoe/xps-autofit
```

---

*Last updated: April 2026*
