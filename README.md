# AI 복습 퀴즈 & 학습 가이드 생성기

강의 스크립트(STT)와 커리큘럼을 기반으로 퀴즈와 주차별 학습 가이드를 자동 생성하는 웹 서비스입니다.

---

## 요구 사항

- **Python 3.11** (3.10 이상 권장)
- 강의 스크립트 텍스트 파일 및 `강의 커리큘럼.csv` (프로젝트에 포함)

---

## 1. 저장소 클론 후 이동

```bash
git clone <저장소 URL>
cd create_quiz_guide
```

---

## 2. 가상환경 생성 및 활성화

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

실행 정책 오류 시:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

가상환경이 활성화되면 프롬프트 앞에 `(venv)`가 표시됩니다.

---

## 3. 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 4. 환경 변수 설정 (.env)

프로젝트 루트에 `.env` 파일을 만들고 OpenAI API 키를 넣습니다.

```bash
# .env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

- **퀴즈/학습 가이드 생성**은 OpenAI Chat API(GPT-4o)를 사용합니다.
- **벡터DB 임베딩**은 `config.yaml`에서 `embedding_backend: "local"`로 두면 API 없이 로컬 모델만 사용할 수 있습니다 (할당량/비용 절약).

`.env`는 Git에 올리지 마세요. (이미 `.gitignore`에 포함하는 것을 권장합니다.)

---

## 5. 설정 확인 (config.yaml)

- **embedding_backend**: `"local"` (로컬 임베딩, API 불필요) 또는 `"openai"` (OpenAI 임베딩 API 사용)
- **paths**: `강의 스크립트` 폴더, `강의 커리큘럼.csv` 경로가 프로젝트 구조와 맞는지 확인

필요 시 `config.yaml`에서 위 항목만 수정하면 됩니다.

---

## 6. Streamlit 앱 실행

```bash
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 터미널에 표시되는 주소(예: `http://localhost:8501`)로 접속합니다.

---

## 7. 최초 실행 시: 벡터DB 구축

1. 메인 페이지에서 **「벡터DB 구축하기」** 버튼을 클릭합니다.
2. 강의 스크립트를 분석해 임베딩을 만들고 FAISS 인덱스를 저장합니다.
3. 한 번 구축해 두면 이후에는 자동으로 저장된 벡터DB를 사용합니다.

- `embedding_backend: "local"`이면 최초 1회 로컬 모델 다운로드가 있을 수 있습니다.
- 프로젝트 경로에 한글이 있으면 벡터DB는 시스템 임시 폴더에 저장되며, 경로는 `data/vectorstore/faiss_location.txt`에 기록됩니다.

---

## 8. 주요 기능

| 페이지 | 설명 |
|--------|------|
| 메인 | 벡터DB 구축, 강의 일정 확인, 각 페이지로 이동 |
| 퀴즈 풀기 | 날짜/유형/난이도 선택 후 퀴즈 생성 및 풀이, 정답/해설 확인 |
| 학습 가이드 | 주차별·날짜별 요약, 핵심 개념, 복습 포인트, 개념 관계 맵 |
| 학습 분석 | 퀴즈 결과 통계, 취약 영역, 학습 추천 |

---

## 9. 디렉터리 구조 (참고)

```
create_quiz_guide/
├── app.py                 # Streamlit 메인 앱
├── config.yaml            # 설정 (임베딩 백엔드, 경로 등)
├── requirements.txt
├── .env                   # OPENAI_API_KEY (본인 PC에서만 생성)
├── src/                   # 핵심 모듈
├── pages/                 # 퀴즈 풀기, 학습 가이드, 학습 분석
├── data/
│   ├── vectorstore/       # FAISS 인덱스 및 메타데이터 (구축 후 생성)
│   └── generated/         # 생성된 퀴즈·가이드 JSON 캐시
├── 강의 스크립트/          # STT 텍스트 파일 (.txt)
└── 강의 커리큘럼.csv       # 주차·날짜·과목·학습목표
```

---

## 10. 문제 해결

| 현상 | 확인 사항 |
|------|------------|
| `pip` 또는 `streamlit`을 찾을 수 없음 | 가상환경이 활성화되었는지 확인 (`(venv)` 표시). `python -m pip`, `python -m streamlit run app.py` 사용 |
| OpenAI 429 (할당량 초과) | 퀴즈/가이드 생성은 API 필요. `config.yaml`의 `embedding_backend`를 `"local"`로 두면 **벡터DB 구축**만 API 없이 가능 |
| 벡터DB 구축 시 "Illegal byte sequence" | 프로젝트 경로에 한글이 있어도 코드에서 임시 경로로 저장하도록 되어 있음. 최신 코드 기준으로 다시 시도 |
| 퀴즈/가이드가 생성되지 않음 | `.env`에 `OPENAI_API_KEY`가 올바르게 설정되었는지, 네트워크/방화벽 확인 |

---

## 11. 팀원 공유 체크리스트

- [ ] 저장소 클론
- [ ] Python 3.11(또는 3.10+) 설치
- [ ] 가상환경 생성 및 활성화
- [ ] `pip install -r requirements.txt`
- [ ] `.env`에 `OPENAI_API_KEY` 설정 (퀴즈/가이드 생성용)
- [ ] `streamlit run app.py` 실행
- [ ] 메인 페이지에서 벡터DB 구축 1회 실행

위 순서대로 진행하면 동일 환경에서 구동할 수 있습니다.
