# XPS AutoFit — 실행 & 배포 가이드

본인은 코드를 쓸 필요 없습니다. **복사/붙여넣기**만 하면 됩니다.

---

## 📦 이 폴더에 들어있는 파일

| 파일 | 역할 |
|---|---|
| `xps_engine.py` | 피팅 코어 엔진 (Shirley, Voigt, AIC 모델 선택) |
| `app.py` | Streamlit 웹앱 UI |
| `requirements.txt` | Python 의존성 |
| `README.md` | 이 문서 |

---

## 🖥️ Step 1: 로컬에서 실행해보기 (5분)

### Windows (Anaconda 권장)

```bash
# 1. Anaconda Prompt 열기
# 2. 폴더로 이동
cd 경로/XPS_AutoFit

# 3. 필수 라이브러리 설치 (한 번만)
pip install -r requirements.txt

# 4. 앱 실행
streamlit run app.py
```

브라우저가 자동으로 열리고 `http://localhost:8501`에 앱이 뜹니다.

### macOS / Linux

```bash
cd 경로/XPS_AutoFit
pip3 install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Step 2: 테스트

1. 웹 UI에서 사이드바의 **"CSV 또는 TXT 파일"** 클릭
2. 업로드했던 `1.csv`, `2.csv`, `3.csv`, `4.csv` 중 하나 선택
   - (원본이 .xls면 Excel에서 `다른 이름으로 저장` → `CSV(.csv)`)
3. 자동 피팅 결과 확인
4. 좌측 사이드바에서 **"피크 수 수동 지정"** 바꿔서 영향 관찰
5. 하단 **다운로드 버튼**으로 결과 저장

---

## 🚀 Step 3: 웹에 배포하기 — Streamlit Cloud (무료, 가장 쉬움)

### 3-1. GitHub 준비
1. GitHub 계정 (없으면 github.com에서 가입)
2. 새 저장소 생성 → 이름: `xps-autofit` (private 가능)
3. 이 폴더의 **파일 4개** (`app.py`, `xps_engine.py`, `requirements.txt`, `README.md`) 업로드
   - 초보자면 **GitHub Desktop** 앱 쓰면 드래그&드롭 가능

### 3-2. Streamlit Cloud 배포
1. https://share.streamlit.io 접속 → GitHub으로 로그인
2. `New app` 클릭
3. 폼 작성:
   - Repository: `본인아이디/xps-autofit`
   - Branch: `main`
   - Main file path: `app.py`
4. `Deploy` 클릭
5. 1~2분 후 `https://본인아이디-xps-autofit.streamlit.app` 같은 URL이 나옴
6. **그 URL을 누구에게든 공유 가능**

> 💡 **Tip**: 무료 플랜은 30일 inactive시 sleep. 한 달에 한 번 열면 계속 유지됨. 도메인 연결, password 보호는 유료. 초기엔 무료면 충분.

---

## 🛠️ Step 4 (선택): Render.com 배포 — 더 유연한 옵션

Streamlit Cloud가 마음에 안 들면:

1. https://render.com 가입 (GitHub 연동)
2. `New +` → `Web Service` → GitHub 저장소 연결
3. 설정:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - Plan: **Free**
4. 배포 완료 → `https://xps-autofit.onrender.com` 같은 URL

> ⚠️ Render 무료 티어는 15분 inactive시 sleep, 첫 요청이 느림(cold start ~30초).

---

## 🐛 문제 해결

| 증상 | 해결 |
|---|---|
| `streamlit: command not found` | `pip install streamlit` 다시 |
| `ModuleNotFoundError: xps_engine` | `app.py`와 `xps_engine.py`가 같은 폴더인지 확인 |
| 한글 깨짐 | CSV 저장 시 UTF-8 인코딩 선택 |
| 업로드 후 "데이터 행 없음" | CSV 구조 확인. BE는 1번째 열, Counts는 3번째 열(CasaXPS 기본) |
| 배포 앱이 안 뜸 | Streamlit Cloud 로그 탭 확인 (`Manage app` → `Logs`) |

---

## 📝 앞으로의 로드맵 (2~4주차)

- **2주차**: 스핀-오빗 doublet 자동 처리 (Sn 3d, Cu 2p 등)
- **3주차**: NIST XPS DB 기반 화학상태 라이브러리 매칭
- **4주차**: Multi-file 비교, constraint UI (peak area ratio lock 등)

---

## ⚠️ 사용 시 주의사항

1. **자동 피팅 결과는 절대 "진리"가 아닙니다.** 화학자/본인의 도메인 지식으로 검증하세요.
2. **R² 높다고 옳은 모델 아닙니다.** AIC, residual 패턴, 물리적 합당성을 함께 보세요.
3. **F1s (s-orbital)**만 우선 최적화됨. 3d, 2p 같은 doublet 데이터는 Phase 1에서 지원 예정.
4. 업로드된 데이터는 **서버 메모리에서만 처리**되고 저장되지 않음.
