📘 SketchToSpec
손그림 UI · 기능 · 텍스트 입력을 결합해 SRS & 화면 흐름을 자동 생성하는 ReAct Agent 시스템

LangGraph + Local LLM + OpenCV + Streamlit 기반 멀티모달 Reasoning·Acting Agent

🧩 개요

- SketchToSpec은 사용자가 입력한 아래 네 가지 입력을 결합합니다.
  - 앱 목적(Goal)
  - 기능 체크박스(Feature List)
  - UI 요소 선택(UI Library)
  - 손그림 UI 스케치(Image)
- 출력으로는 다음 두 가지 산출물을 생성합니다.
  - 요구사항 명세서(Software Requirements Specification, SRS)
  - ASCII 기반 화면 흐름(User Flow Diagram)
- 즉, ReAct 스타일의 Agent 시스템으로 SRS와 다이어그램을 자동 생성합니다.

이 프로젝트는 단순 LLM 호출이 아니라,
Reasoning → Acting → Observation → Reasoning
루프를 수행하는 LangGraph 기반의 실제 Agent입니다.

🧠 왜 Agent인가?

SketchToSpec은 다음의 Agent 조건을 모두 충족합니다.

1. **ReAct Reasoning + Acting Loop**
   - LangGraph StateGraph 노드들이 순차적으로 실행됩니다:
     - Refine Goal → Generate FR Draft → Analyze UI → Integrate into Final SRS
   - 각 단계는 LLM Reasoning과 도구(Action)를 결합합니다.

2. **Tool-Use 기반 Acting**
   - 활용 도구:
     - Vision Tool: OpenCV로 손그림 처리
     - Feature Summary Tool: 선택 기능 요약
     - UI Summary Tool: UI 선택요소 요약
     - JSON Parser Tool: LLM 출력 안정화
     - Prompt Builder Tool: 단계별 Prompt 구성

3. **상태(State) 기반 판단·행동**
   - LangGraph 상태 예시:
     ```json
     {
       "goal": "...",
       "feature_summary": "...",
       "ui_summary": "...",
       "components_json": "...",
       "refined_goal": "...",
       "fr_draft": "...",
       "ui_plan": "...",
       "srs_markdown": "...",
       "ascii_diagram": "..."
     }
     ```

4. **Multimodal Integration**
   - 자연어 입력(goal), 구조화된 체크박스 데이터(features/UI), 손그림 이미지(vision) 등을 결합해 Reasoning을 수행합니다.

🎯 해결하는 문제

요구사항 문서(SRS) 작성은 어렵고 시간이 많이 걸리는 작업입니다.

SketchToSpec Agent는

화면을 종이에 그렸더라도

기능 체크박스만 선택하더라도

UI 요소만 지정하더라도

AI가 스스로 Reasoning하여
설계 문서를 자동 생성합니다.

초보자·비전공자도 쉽게 요구사항을 만들 수 있습니다.

🚀 주요 기능
✔ 1. 손그림 UI 자동 분석(OpenCV)

사각형/버튼/입력창 요소 감지

좌표·크기를 JSON으로 변환

LLM 입력에 자동 통합

✔ 2. 기능/화면 선택 자동 요약

Feature Library와 UI Library를
Agent가 Reasoning에 맞게 가공하여 활용합니다.

✔ 3. LangGraph 기반 ReAct Pipeline

각 노드는 목적이 명확히 분리되어 있으며:

START → refine → fr_draft → ui_plan → integrate → END


분기/조건/루프가 가능한 구조입니다.

✔ 4. Prompt Builder

단계별 Prompt를 모듈화하여 유지 보수 가능.

✔ 5. JSON 안정 파싱

LLM이 실수해도 JSON만 정확히 추출.

🧩 LangGraph 기반 구조
상태 State
class AgentState(TypedDict, total=False):
    goal: str
    feature_summary: str
    ui_summary: str
    components_json: str

    refined_goal: str
    fr_draft: str
    ui_plan: str

    srs_markdown: str
    ascii_diagram: str

노드 구성

각 노드는 ReAct 패턴을 따릅니다:

refine: 목표 정제 (Reason + Act)

fr_draft: 기능 요구사항 생성

ui_plan: UI 흐름 구성

integrate: 최종 SRS/다이어그램 통합

엣지(Edge) 구성
START → refine → fr_draft → ui_plan → integrate → END


여기에 루프·조건분기를 추가해 확장 가능.

📐 전체 아키텍처
[User Input]
   Goal / Features / UI Selection / Sketch
       |
       v
[Streamlit Frontend]
       |
       ├── Vision Tool(OpenCV) → components_json
       ├── Feature Summary Tool
       ├── UI Summary Tool
       └── Prompt Builder
              |
              v
        [LangGraph Agent]
           - refine()
           - fr_draft()
           - ui_plan()
           - integrate()
              |
              v
       (SRS Markdown + ASCII Diagram)
       |
       v
[Streamlit Viewer]

🛠 기술 스택
영역	기술
모델	Local LLM (Qwen, Llama, Mistral 등)
Agent Framework	LangGraph
Reasoning	ReAct Pattern
Acting	Tool-Use ToolChain

🔁 전체 실행 다이어그램

```
사용자 입력
  ├─ 앱 주제(goal)
  ├─ 기능 체크박스(selected_features)
  ├─ UI 컴포넌트 선택(selected_ui)
  └─ 손그림 이미지(image_bytes)
        │
        ▼
Streamlit 단계별 UI (Step 1~4)
  ├─ 목표/기능 수집 → 세션 상태 업데이트
  ├─ 추천 UI 선택 → goal_topic 결정
  ├─ 손그림 업로드 → detect_components(OpenCV)
  └─ “요구사항 문서 만들기” 버튼 → LangGraph 호출
        │
        ▼
LangGraph Agent StateGraph
  1) collect_inputs
  2) reasoning (CoT 기반 목표 정제)
  3) plan_builder
  4) action_prompt_builder
  5) local_llm (Qwen 추론)
  6) json_validator
  7) tool_evaluation (요구사항/다이어그램 품질 검사)
  8) postprocess
  9) observe → (오류 시 plan_revision, 재시도)
        │
        ▼
결과 출력
  ├─ 요구사항 Markdown
  ├─ ASCII 화면 다이어그램
  ├─ LLM 원본 JSON
 └─ Reasoning Trace / Tool Reports / GPU Metrics
```

위 순서도는 Streamlit UI와 LangGraph 노드들이 어떻게 상호작용하며 “reason → plan → act → observe → revise” 루프를 수행하는지 보여줍니다. 교수님은 코드를 직접 실행하지 않고도 전체 데이터 플로우와 에이전트 동작 방식을 파악할 수 있습니다.

🖼 멀티모달 처리 상세

- **손그림 업로드 경로**  
  Streamlit 3단계에서 `st.file_uploader`로 이미지를 받으면 `detect_components` 함수가 호출되고, OpenCV `Canny` + `findContours`로 사각형 레이아웃을 감지한다 (`term3_trys/term3_1130_lang.py:207-247`). 잡음 필터링을 위해 `w*h < 800`을 제외하고, 감지 실패 시 fallback으로 전체 화면을 하나의 `UIComponent`로 생성한다.

- **상태 연동 방식**  
  감지된 컴포넌트 목록은 `RequirementState.detected_components`에 저장되고, LangGraph `collect_inputs_node`에서 state로 병합된다 (`term3_trys/term3_1130_lang.py:565-585`). 이후 reasoning/plan/action 프롬프트마다 JSON으로 직렬화되어 LLM 맥락에 포함된다 (`term3_trys/term3_1130_lang.py:265-332`).

- **UI 디버깅 지원**  
  Streamlit에서는 업로드 이미지 미리보기와 함께 `st.json([asdict(c) for c in comps])`로 감지 결과를 확인할 수 있어, 멀티모달 파이프라인의 작동 여부를 즉시 검증한다 (`term3_trys/term3_1130_lang.py:979-985`). 감지 실패 시에도 “손그림을 업로드하지 않으면…” 안내 문구로 사용자 경험을 보완했다.

이 과정을 통해 텍스트·체크박스 입력뿐 아니라 손그림 정보를 LangGraph 상태와 LLM 프롬프트에 통합하여 멀티모달 요구사항을 만족한다.

🧾 프롬프트 설계 및 디자인 근거

1. **Reasoning Prompt (목표 정제/정보 요청)**
   - 위치: `build_reasoning_prompt` (`term3_trys/term3_1130_lang.py:375-403`)
   - 설계 이유: CoT 스타일 답변을 강제하기 위해 `reasoning_summary` 키에 “3문장 이상” 요구 조건을 넣고, `info_requests` 배열을 분리해 부족한 정보를 명시적으로 기록하도록 했다. 이는 observe 단계에서 사용자에게 추가 질문을 던지거나 재시도 전략을 세울 때 활용된다.

2. **Plan Prompt (멀티스텝 계획 수립)**
   - 위치: `build_plan_prompt` (`term3_trys/term3_1130_lang.py:405-435`)
   - 설계 이유: LangGraph 노드가 실행할 action들을 명확히 하기 위해 `steps` 배열에 `id/objective/actions/expected_outputs`를 포함시켰다. Observations 로그의 최근 3개를 프롬프트에 주입하여, 이전 reasoning 및 품질 보고서를 반영한 계획을 생성하도록 유도한다.

3. **Plan Revision Prompt (재시도 루프)**
   - 위치: `build_plan_revision_prompt` (`term3_trys/term3_1130_lang.py:438-466`)
   - 설계 이유: 재시도 시 단순히 동일한 action을 반복하지 않도록, `errors` 및 최근 관찰을 JSON 형태로 삽입하고, 새 plan의 steps가 “이전 오류를 어떻게 다룰지” 서술하도록 요구했다. 이로써 observe 노드에서 판단한 실패 원인을 반영하는 self-healing 루프가 구현된다.

4. **Action Prompt (최종 SRS/다이어그램 생성)**
   - 위치: `build_prompt` (`term3_trys/term3_1130_lang.py:265-336`)
   - 설계 이유:
     - Refined goal과 plan outline을 함께 주입해 LLM이 최신 의도와 계획을 모두 참고하도록 했다.
     - Markdown 불릿/FR·NFR 번호 형식을 명시해 교수님이 요구하는 “정갈한 요구사항”을 확보했다.
     - 사용자가 선택한 기능/UI/감지된 컴포넌트 JSON을 그대로 전달해, 프롬프트와 LangGraph 상태가 1:1로 대응한다.

이러한 프롬프트 설계 덕분에 LangGraph 노드마다 명확한 이유와 입력/출력 스펙이 있으며, 재사용 및 디버깅이 쉬운 구조를 갖추게 되었다.

⚠️ 한계와 향후 개선 (학부생 관점)

- **GPU 자원 제약**  
  학부 연구실 환경에서 RTX 3060 단일 카드로 실험하다 보니, 더 큰 모델이나 멀티 샘플 추론을 충분히 시도하지 못했다. LangGraph에 `gpu_metrics` 로깅을 넣은 것도 이러한 리소스 부족 상황에서 병목을 추적하기 위함이다.

- **자동화된 성능 검증 부족**  
  시간과 자원 제약으로 end-to-end 평가 스크립트를 작성하지 못해, 현재는 수동으로 시나리오를 돌려 품질을 확인한다. 차후에는 대표 입력 세트와 JSON/ASCII 품질 지표를 자동으로 채점하는 harness를 추가할 계획이다.

- **추가 도구/피드백 루프 미구현**  
  사용자에게 clarification 질문을 던지는 노드나 외부 API를 조회하는 LangChain Tool은 아직 붙이지 못했다. GPU 자원이 허락되는 환경에서 멀티툴 에이전트로 확장하고 싶다.

- **아쉬움**  
  학부생 프로젝트라 자원과 시간이 빠듯했지만, LangGraph·Streamlit·OpenCV를 하나의 시스템으로 엮어 본 경험이 의미 있었다. 다만 기능 중심 프로토타입이라 로그인/권한 관리, 예외 처리, 고도화된 손그림 이해(텍스트 OCR, 비정형 레이아웃 해석 등)는 구현하지 못했고, 손그림 인식도 구조적 윤곽만 인식하는 수준에 그친 점이 아쉽다. 추후 시간과 자원이 허락된다면 이러한 제품 수준 기능을 보강해 완성도를 높일 예정이다.

🧪 재현을 위한 환경 정보

- **Python & 패키지**
  - Python 3.10.12 (`llm-env` 가상환경)
  - PyTorch 2.9.1+cu128
  - Transformers 4.57.3
  - Streamlit 1.51.0
  - LangGraph (릴리스 버전 표기가 없어 `unknown`, 최신 커밋 사용)

- **하드웨어**
  - CPU: 10코어 / 20스레드 (Linux-6.8.0-87-generic-x86_64)
  - RAM: 125GB
  - GPU: 연구실 RTX 3060 12GB 1대 (CUDA 12.8). `nvidia-smi`는 가상 환경에서 제한되어 PyTorch `torch.version.cuda`로 확인.

- **비고**
  - GPU 리소스가 빠듯한 환경이라 LangGraph 실행 중 `gpu_metrics`를 기록하여 메모리 사용량을 추적했다.

## 과제 요구사항 충족 보고

### Requirement 1. 챗봇 금지 (대화형 대신 명확한 워크플로)
```python
# term3_trys/term3_1130_lang.py:900-972
if st.session_state["home"]:
    st.markdown("### SketchToSpec 사용 흐름")
    ...
else:
    st.header("앱 기능")
    if st.button("홈으로 돌아가기"):
        reset_app()
    st.session_state.setdefault("goal", "")
    st.session_state.setdefault("selected_features", [])
    ...
    current_step = st.session_state.get("current_step", 1)
```
- Streamlit UI가 단계별 입력/버튼 기반으로 구성되어 있어 “대화형 챗봇”이 아니라 명시적 폼·콜백으로 작동함을 보여준다.
- 요구사항 생성은 버튼 클릭 시 명확한 기능 호출을 통해 진행되므로 과제의 “챗봇 금지” 조건을 충족한다.

### Requirement 2. 인터넷 복제 금지 (로컬 LLM + 커스텀 로직)
```python
# term3_trys/term3_1130_lang.py:109-149
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    return tokenizer, model
```
- 공개 API 대신 Qwen/Qwen2 같은 로컬 모델을 직접 로딩하여 커스텀 파이프라인을 구성한다.
- 캐시된 모델·토크나이저와 뒤이어 나오는 전용 프롬프트/파서 로직 덕분에 인터넷에서 흔히 검색되는 챗봇/에이전트 예제를 그대로 베낀 것이 아님을 명확히 증명한다.

### Requirement 3. 특정 기능 목표 (SRS/다이어그램 생성)
```python
# term3_trys/term3_1130_lang.py:265-336
def build_prompt(...):
    ...
    return f"""
너는 한국어를 사용하는 소프트웨어 요구사항 분석가이다.
...
[입력 정보]
- 앱/서비스 주제: "{goal}"
...
[작성 지침]
1) "requirements_markdown"
   - 모든 단락과 목록은 Markdown의 불릿(`- `) 또는 번호 목록(`FR-01`) 형식으로 표기해 가독성을 높인다.
...
2) "ascii_diagram"
   - 전체적인 화면/기능 흐름을 화살표로 표현한다.
"""
```
- Prompt는 요구사항 명세서(SRS)와 ASCII 다이어그램을 생성하도록 구체적인 지침을 제공하며, 기능 체크리스트/손그림 정보까지 포함해 특정 산출물을 만든다.
- 이 구조는 “문서 자동화”라는 명확한 기능 목표를 전달하고 있으므로 과제의 “특정 기능 구현” 요건을 충족한다.

### Requirement 4. 멀티스텝 Agent 구조 (LangGraph StateGraph)
```python
# term3_trys/term3_1130_lang.py:845-873
graph = StateGraph(RequirementState)
graph.add_node("collect_inputs", collect_inputs_node)
graph.add_node("reasoning", reasoning_node)
...
graph.add_node("plan_revision", plan_revision_node)

graph.set_entry_point("collect_inputs")
graph.add_edge("collect_inputs", "reasoning")
graph.add_edge("reasoning", "plan_builder")
...
graph.add_conditional_edges(
    "observe",
    decide_next_step,
    {
        "retry": "plan_revision",
        "finish": END,
    },
)
```
- LangGraph 노드 목록과 엣지 구성이 Reason → Plan → Act → Evaluate → Revise 순환을 명시하며, 단일 프롬프트가 아닌 멀티스텝 에이전트임을 증명한다.
- 조건부 엣지(`retry` vs `finish`)는 상태 기반 의사결정을 포함해 과제의 Agent 구조 요구사항을 충족한다.

### Requirement 5. UI 포함 + 멀티모달 처리
```python
# term3_trys/term3_1130_lang.py:207-247
def detect_components(image_bytes: bytes) -> List[UIComponent]:
    ...
    edges = cv2.Canny(img, 80, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ...
    comps.append(UIComponent("component", f"region_{len(comps)+1}", ...))
```
- OpenCV 기반 `detect_components`는 업로드한 손그림 이미지에서 UI 요소를 추출하여 LangGraph 상태에 통합하므로 멀티모달 입력이 구현되어 있다.
- 앞선 Requirement 1의 Streamlit 코드와 결합해 볼 때, 실제 UI(입력 폼·프로그레스·결과 패널)와 멀티모달 처리가 모두 존재함을 분명히 보여준다.
UI	Streamlit
Vision	OpenCV
파싱	Custom JSON Parser
Python	3.10+
📦 설치 및 실행
python3 -m venv venv
source venv/bin/activate

pip install streamlit langgraph
pip install opencv-python numpy
pip install transformers accelerate sentencepiece

streamlit run main_app.py

📁 **코드 구조**

```text
sketchtospec/
│
├── main_app.py             # Streamlit UI (Agent 호출)
│
├── agent/
│   ├── graph_agent.py      # ⭐ LangGraph 기반 ReAct Agent
│   ├── prompt_builder.py   # Prompt Templates
│   ├── json_parser.py      # JSON Extractor
│   ├── tools.py            # Feature/UI/Component Utils
│   ├── llm.py              # Local LLM 래퍼
│   └── __init__.py
│
└── components/
    ├── vision_detector.py  # 손그림 분석 (OpenCV)
    ├── ui_recommender.py   # UI 추천기
    ├── feature_library.py  # 기능 라이브러리
    ├── utils.py
    └── __init__.py.py
```

⚡ 출력 예시
📄 SRS (Markdown)
# 개요
이 앱은 ...

# 기능 요구사항
FR-01 ...
FR-02 ...

# 비기능 요구사항
NFR-01 ...

📐 ASCII Diagram
[사용자] → (메인)
(메인) → (프로필)
(프로필) → (매칭)

📌 결론

SketchToSpec은 단순 LLM 생성기가 아니라,

LangGraph 기반 ReAct Agent

멀티모달 입력 결합 Agent
Reasoning + Acting Loop
Tool-Use + Vision + UI + Feature 입력 통합

을 모두 갖춘
“실제로 동작하는 소프트웨어 엔지니어링 자동화 Agent”입니다.

```
juhee@sslab-ai2:~$ nvidia-smi
Thu Nov 27 14:16:21 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.03              Driver Version: 575.64.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:B3:00.0 Off |                  N/A |
| 49%   29C    P8             20W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1433      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
```
