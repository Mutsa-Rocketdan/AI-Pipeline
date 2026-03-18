"""퀴즈 유형별/난이도별 프롬프트 템플릿 모듈."""

SYSTEM_PROMPT = """당신은 IT 교육 콘텐츠 전문가입니다.
강의 스크립트(STT 기반 텍스트)를 분석하여 수강생의 학습을 돕는 고품질 퀴즈와 학습 자료를 생성합니다.

중요 지침:
- 제공된 텍스트는 STT(음성→텍스트) 변환 결과이므로 오탈자나 구어체 표현이 있을 수 있습니다. 의미를 정확히 파악하여 활용하세요.
- 퀴즈는 강의에서 실제로 다룬 내용만을 기반으로 생성하세요.
- 모든 응답은 한국어로 작성하세요.
- 반드시 지정된 JSON 형식으로만 응답하세요."""

DIFFICULTY_GUIDE = {
    "easy": "기본 정의, 용어 설명, 단순 사실 확인 수준의 문제를 출제하세요.",
    "medium": "개념 간 비교, 차이점 설명, 적용 시나리오 판단 수준의 문제를 출제하세요.",
    "hard": "복합 개념 통합, 코드 분석, 실무 적용 및 문제 해결 수준의 문제를 출제하세요.",
}

QUIZ_MULTIPLE_CHOICE = """아래 강의 내용을 바탕으로 {difficulty} 난이도의 객관식 퀴즈 {count}개를 생성하세요.

[강의 내용]
{context}

[학습 목표]
{learning_goal}

[난이도 가이드]
{difficulty_guide}

[출력 형식]
반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
```json
[
  {{
    "type": "multiple_choice",
    "difficulty": "{difficulty}",
    "question": "문제 내용",
    "options": ["보기1", "보기2", "보기3", "보기4", "보기5"],
    "answer": "정답 보기 텍스트",
    "explanation": "정답에 대한 상세 해설"
  }}
]
```

규칙:
- 보기는 5개로 구성하되, 그럴듯한 오답(distractors)을 포함하세요.
- 정답은 options 리스트에 포함된 텍스트와 정확히 일치해야 합니다.
- 해설은 왜 해당 답이 정답인지, 나머지 보기가 왜 틀린지 설명하세요.
- 강의에서 다루지 않은 내용으로 문제를 만들지 마세요."""

QUIZ_SHORT_ANSWER = """아래 강의 내용을 바탕으로 {difficulty} 난이도의 주관식(단답형) 퀴즈 {count}개를 생성하세요.

[강의 내용]
{context}

[학습 목표]
{learning_goal}

[난이도 가이드]
{difficulty_guide}

[출력 형식]
반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
```json
[
  {{
    "type": "short_answer",
    "difficulty": "{difficulty}",
    "question": "문제 내용",
    "options": null,
    "answer": "모범 답안 (간결한 핵심 키워드/문장)",
    "explanation": "정답에 대한 상세 해설"
  }}
]
```

규칙:
- 답변은 핵심 용어나 짧은 문장으로 구성하세요.
- 해설에서 관련 개념을 충분히 설명하세요."""

QUIZ_FILL_BLANK = """아래 강의 내용을 바탕으로 {difficulty} 난이도의 빈칸 채우기 퀴즈 {count}개를 생성하세요.

[강의 내용]
{context}

[학습 목표]
{learning_goal}

[난이도 가이드]
{difficulty_guide}

[출력 형식]
반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
```json
[
  {{
    "type": "fill_blank",
    "difficulty": "{difficulty}",
    "question": "핵심 문장에서 _____ 부분을 채우시오.",
    "options": null,
    "answer": "빈칸에 들어갈 정답",
    "explanation": "해당 개념에 대한 상세 해설"
  }}
]
```

규칙:
- 문제의 빈칸은 _____로 표시하세요.
- 핵심 용어나 개념이 빈칸이 되도록 하세요.
- 문맥으로 유추할 수 있되 암기가 필요한 수준이어야 합니다."""

QUIZ_CODE = """아래 강의 내용을 바탕으로 {difficulty} 난이도의 코드 관련 퀴즈 {count}개를 생성하세요.
강의 주제에 따라 SQL 쿼리, Java 코드 등 적절한 코드 문제를 출제하세요.

[강의 내용]
{context}

[학습 목표]
{learning_goal}

[난이도 가이드]
{difficulty_guide}

[출력 형식]
반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
```json
[
  {{
    "type": "code",
    "difficulty": "{difficulty}",
    "question": "코드를 포함한 문제 설명\\n```sql\\nSELECT ...\\n```\\n위 쿼리의 실행 결과는?",
    "options": ["결과1", "결과2", "결과3", "결과4", "결과5"],
    "answer": "정답",
    "explanation": "코드 실행 과정과 결과에 대한 상세 해설"
  }}
]
```

규칙:
- 코드 블록은 문제 내에 포함하세요.
- 객관식 또는 단답형 중 적절한 형태로 출제하세요.
- 객관식이면 options를 제공하고, 단답형이면 options를 null로 설정하세요."""

QUIZ_TEMPLATES = {
    "multiple_choice": QUIZ_MULTIPLE_CHOICE,
    "short_answer": QUIZ_SHORT_ANSWER,
    "fill_blank": QUIZ_FILL_BLANK,
    "code": QUIZ_CODE,
}

STUDY_GUIDE_SUMMARY = """아래는 {week}주차 ({dates}) 강의 내용입니다. 주차별 핵심 요약을 작성하세요.

[강의 내용]
{context}

[커리큘럼 정보]
- 과목: {subject}
- 학습 목표: {learning_goal}

[출력 형식]
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
```json
{{
  "weekly_summary": "주차 전체 학습 내용을 3~5문단으로 요약",
  "daily_summaries": [
    {{
      "date": "YYYY-MM-DD",
      "summary": "해당 일자 강의 핵심 내용 요약 (2~3문장)"
    }}
  ],
  "key_concepts": [
    {{
      "term": "핵심 용어/개념",
      "definition": "간결한 정의 설명"
    }}
  ],
  "review_points": [
    "복습 포인트 1: 구체적인 복습 항목",
    "복습 포인트 2: 구체적인 복습 항목"
  ],
  "concept_relations": [
    {{
      "from": "개념A",
      "to": "개념B",
      "relation": "관계 설명 (예: A는 B의 상위 개념)"
    }}
  ]
}}
```

규칙:
- 핵심 개념은 10~15개 추출하세요.
- 복습 포인트는 중요도 순으로 5~10개 나열하세요.
- 개념 관계는 주요 개념 간 관계를 5~8개 정리하세요.
- STT 오류를 보정하여 정확한 용어를 사용하세요."""

STUDY_GUIDE_DAILY = """아래는 {date} 강의 스크립트 내용입니다. 이 날 강의의 핵심 요약을 작성하세요.

[강의 내용]
{context}

[커리큘럼 정보]
- 과목: {subject}
- 내용: {content}
- 학습 목표: {learning_goal}

[출력 형식]
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
```json
{{
  "date": "{date}",
  "summary": "강의 핵심 내용 요약 (3~5문장)",
  "key_concepts": [
    {{
      "term": "핵심 용어/개념",
      "definition": "간결한 정의"
    }}
  ],
  "review_points": ["복습 포인트 1", "복습 포인트 2"]
}}
```"""


def get_quiz_prompt(
    quiz_type: str,
    context: str,
    learning_goal: str,
    difficulty: str = "medium",
    count: int = 5,
) -> str:
    """퀴즈 유형에 맞는 프롬프트 생성."""
    template = QUIZ_TEMPLATES.get(quiz_type, QUIZ_MULTIPLE_CHOICE)
    difficulty_guide = DIFFICULTY_GUIDE.get(difficulty, DIFFICULTY_GUIDE["medium"])

    return template.format(
        context=context,
        learning_goal=learning_goal,
        difficulty=difficulty,
        difficulty_guide=difficulty_guide,
        count=count,
    )


def get_study_guide_prompt(
    context: str,
    week: int,
    dates: str,
    subject: str,
    learning_goal: str,
) -> str:
    """주차별 학습 가이드 생성 프롬프트."""
    return STUDY_GUIDE_SUMMARY.format(
        context=context,
        week=week,
        dates=dates,
        subject=subject,
        learning_goal=learning_goal,
    )


def get_daily_guide_prompt(
    context: str,
    date: str,
    subject: str,
    content: str,
    learning_goal: str,
) -> str:
    """일별 학습 가이드 생성 프롬프트."""
    return STUDY_GUIDE_DAILY.format(
        context=context,
        date=date,
        subject=subject,
        content=content,
        learning_goal=learning_goal,
    )
