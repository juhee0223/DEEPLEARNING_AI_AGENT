#!/usr/bin/env python3
"""
SketchToSpec: 주제 + 기능 체크박스 + 손그림 → 요구사항 & ASCII 다이어그램
- Streamlit UI
- 로컬 GPU LLM 사용 (예: Qwen/Qwen2-7B-Instruct)
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, TypedDict, Optional

import streamlit as st

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    cv2 = None
    np = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langgraph.graph import StateGraph, END

# ------------------------------------------------------------------------------------
# 모델 설정
# ------------------------------------------------------------------------------------
MODEL_NAME = os.getenv("SKETCHTOSPEC_MODEL", "Qwen/Qwen2-7B-Instruct")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.4
TOP_P = 0.9
MAX_AGENT_RETRIES = 2


# ------------------------------------------------------------------------------------
# Dataclass 정의
# ------------------------------------------------------------------------------------
@dataclass
class UIComponent:
    kind: str
    label: str
    bbox: List[int]  # [x1, y1, x2, y2]


class RequirementState(TypedDict, total=False):
    goal: str
    goal_topic: str
    sanitized_goal: str
    selected_features: List[Dict[str, str]]
    selected_ui: List[Dict[str, str]]
    detected_components: List[UIComponent]
    reasoning_notes: str
    info_requests: List[str]
    plan_outline: Dict[str, Any]
    plan_text: str
    action_prompt: str
    llm_output: str
    parsed_json: Dict[str, Any]
    result_payload: Dict[str, Any]
    errors: List[str]
    observations: List[str]
    actions_taken: List[str]
    tool_reports: List[Dict[str, Any]]
    gpu_metrics: List[Dict[str, Any]]
    retry_count: int
    should_retry: bool


# ------------------------------------------------------------------------------------
# 공통 "기능 체크리스트" 정의 (주제와 무관하게 자주 등장하는 기능)
# ------------------------------------------------------------------------------------
FEATURE_LIBRARY: List[Dict[str, str]] = [
    # 인증 / 계정
    {"key": "signup", "category": "인증/계정", "name": "회원가입", "desc": "이메일/비밀번호로 계정을 생성"},
    {"key": "login", "category": "인증/계정", "name": "로그인", "desc": "기존 계정으로 로그인"},
    {"key": "social_login", "category": "인증/계정", "name": "소셜 로그인", "desc": "카카오/구글 등 외부 계정으로 로그인"},
    {"key": "profile", "category": "인증/계정", "name": "프로필 관리", "desc": "내 정보 보기 및 수정"},

    # 콘텐츠 / 목록
    {"key": "list_view", "category": "콘텐츠", "name": "목록 화면", "desc": "여러 개의 항목을 리스트/카드로 보여줌"},
    {"key": "detail_view", "category": "콘텐츠", "name": "상세 화면", "desc": "선택한 항목의 상세 정보 화면"},

    # 검색 / 필터
    {"key": "search", "category": "검색/필터", "name": "검색", "desc": "키워드로 항목 검색"},
    {"key": "filter", "category": "검색/필터", "name": "필터/정렬", "desc": "조건에 따라 결과를 필터링/정렬"},

    # 커뮤니케이션
    {"key": "chat", "category": "커뮤니케이션", "name": "1:1 채팅", "desc": "사용자 간 대화 기능"},
    {"key": "notification", "category": "커뮤니케이션", "name": "알림", "desc": "새 매칭/메시지 등 알림"},

    # 추천
    {"key": "personalized_feed", "category": "추천", "name": "개인화 피드", "desc": "사용자 선호도에 기반한 콘텐츠 추천"},
    {"key": "trending", "category": "추천", "name": "트렌딩 콘텐츠", "desc": "현재 인기 있는 콘텐츠 표시"},

    # 소셜
    {"key": "user_follow", "category": "소셜", "name": "사용자 팔로우", "desc": "다른 사용자를 팔로우하고 업데이트를 받기"},
    {"key": "comments", "category": "소셜", "name": "댓글", "desc": "콘텐츠에 댓글을 달고 소통"},

    # 운영 / 관리
    {"key": "analytics", "category": "운영/관리", "name": "분석 대시보드", "desc": "사용자 활동 및 성과 분석"},
    {"key": "content_moderation", "category": "운영/관리", "name": "콘텐츠 관리", "desc": "부적절한 콘텐츠를 검토 및 관리"},

    # 기타
    {"key": "dark_mode", "category": "기타", "name": "다크 모드", "desc": "어두운 테마로 전환"},
    {"key": "multi_language", "category": "기타", "name": "다국어 지원", "desc": "여러 언어로 앱 사용 가능"},
]


# ------------------------------------------------------------------------------------
# 모델 로딩
# ------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    return tokenizer, model


def apply_chat_template(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}\n<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


def model_generate(prompt: str, tokenizer, model) -> str:
    messages = [{"role": "user", "content": prompt}]
    chat = apply_chat_template(tokenizer, messages)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 입력 길이만큼은 프롬프트이므로 잘라낸다
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output[0][input_len:]

    # 새로 생성된 부분만 디코딩
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return decoded



def classify_goal(goal: str) -> str:
    """앱 주제를 간단한 카테고리로 분류한다."""
    g = goal.lower()

    # 소개팅 / 매칭
    if any(k in g for k in ["소개팅", "데이트", "매칭", "연애"]):
        return "소개팅/매칭"

    # 예약 / 스케줄링
    if any(k in g for k in ["예약", "스케줄", "예약 시스템", "booking", "book"]):
        return "예약/스케줄링"

    # 쇼핑 / 커머스
    if any(k in g for k in ["쇼핑", "커머스", "몰", "스토어", "store", "shop", "commerce"]):
        return "쇼핑/커머스"

    # 그 외는 공통 패턴
    return "기타(범용)"


# ------------------------------------------------------------------------------------
# 주제 기반 UI 컴포넌트 추천
# ------------------------------------------------------------------------------------
def recommend_components(goal: str) -> Tuple[str, List[Dict[str, str]]]:
    """앱 주제를 분류하고, 해당 주제에 맞는 UI 컴포넌트를 추천한다."""
    topic = classify_goal(goal)

    # 소개팅 / 매칭 서비스
    if topic == "소개팅/매칭":
        recs = [
            {"name": "프로필 카드", "desc": "사용자 정보를 카드 형태로 보여주는 영역"},
            {"name": "매칭 목록 화면", "desc": "추천/매칭된 사람들을 리스트로 보여주는 화면"},
            {"name": "채팅 버튼", "desc": "선택한 사용자와 대화를 시작하는 버튼"},
            {"name": "좋아요 버튼", "desc": "관심을 표시하는 하트/좋아요 버튼"},
            {"name": "필터 바", "desc": "나이/지역 등 검색 조건을 고르는 영역"},
            {"name": "하단 탭바", "desc": "홈/탐색/채팅/마이페이지로 이동하는 네비게이션"},
        ]

    # 예약 서비스
    elif topic == "예약/스케줄링":
        recs = [
            {"name": "캘린더", "desc": "예약 날짜를 고르는 캘린더 UI"},
            {"name": "시간 선택 영역", "desc": "가능한 시간을 선택하는 버튼/리스트"},
            {"name": "예약 목록 화면", "desc": "내 예약들을 모아서 보여주는 화면"},
            {"name": "예약 상세 카드", "desc": "선택한 예약의 상세 정보를 보여주는 카드"},
            {"name": "확인/취소 버튼", "desc": "예약 생성/변경/취소를 확정하는 버튼"},
        ]

    # 쇼핑 / 커머스
    elif topic == "쇼핑/커머스":
        recs = [
            {"name": "검색창", "desc": "상품/카테고리를 검색하는 입력창"},
            {"name": "상품 카드", "desc": "이미지, 이름, 가격이 들어간 상품 카드"},
            {"name": "상품 상세 화면", "desc": "선택한 상품의 상세 정보 화면"},
            {"name": "장바구니 버튼", "desc": "상품을 장바구니에 담는 버튼"},
            {"name": "결제/주문 버튼", "desc": "결제를 진행하는 버튼"},
            {"name": "카테고리 탭", "desc": "카테고리별로 상품을 나누는 탭"},
        ]

    # 기본 추천 (기타/범용)
    else:
        recs = [
            {"name": "상단 제목바", "desc": "화면 제목이 들어가는 영역"},
            {"name": "텍스트 입력창", "desc": "검색/입력에 사용하는 기본 입력창"},
            {"name": "확인 버튼", "desc": "주요 동작을 수행하는 기본 버튼"},
            {"name": "리스트 카드", "desc": "여러 항목을 세로로 나열하는 카드 리스트"},
            {"name": "하단 탭바", "desc": "여러 화면으로 이동하는 공통 네비게이션"},
        ]

    return topic, recs

# ------------------------------------------------------------------------------------
# 손그림 → 간단 컴포넌트 감지
# ------------------------------------------------------------------------------------
def detect_components(image_bytes: bytes) -> List[UIComponent]:
    if cv2 is None or np is None:
        return [UIComponent("screen", "root", [0, 0, 512, 512])]

    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("이미지 디코드 실패")

        edges = cv2.Canny(img, 80, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        comps: List[UIComponent] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 800:
                continue
            comps.append(UIComponent("component", f"region_{len(comps)+1}", [int(x), int(y), int(x + w), int(y + h)]))

        if not comps:
            comps.append(UIComponent("screen", "root", [0, 0, img.shape[1], img.shape[0]]))
        return comps
    except Exception:
        return [UIComponent("screen", "root", [0, 0, 512, 512])]

# ------------------------------------------------------------------------------------
# LLM 프롬프트 생성
# ------------------------------------------------------------------------------------
def build_prompt(
    goal: str,
    goal_topic: str,
    selected_features: List[Dict[str, str]],
    selected_ui: List[Dict[str, str]],
    detected_components: List[UIComponent],
    refined_goal: Optional[str] = None,
    plan_outline: Optional[str] = None,
) -> str:
    feat_json = json.dumps(selected_features, ensure_ascii=False)
    ui_json = json.dumps(selected_ui, ensure_ascii=False)
    comp_json = json.dumps([asdict(c) for c in detected_components], ensure_ascii=False)
    final_goal = refined_goal.strip() if refined_goal else goal

    plan_section = ""
    if plan_outline:
        plan_section = f"""
[현재 계획 요약]
{plan_outline}
"""

    return f"""
너는 한국어를 사용하는 소프트웨어 요구사항 분석가이다.

[역할]
- 사용자가 만들고 싶은 앱의 목적과 기능을 이해하고,
- 핵심 기능 위주의 **상세한** 요구사항 명세서(한국어)를 작성하며,
- 화면 흐름을 간단한 ASCII 다이어그램으로 정리한다.

[추론 맥락]
- 원본 사용자가 입력한 주제: "{goal}"
- 정제된/보정된 목표: "{final_goal}"
- 계획 기반 추론 메모: 위 [현재 계획 요약]을 참고한다.

[출력 형식]
- JSON 한 개만 출력한다.
- 바깥에 설명, 문장, 코드블록, 마크다운을 붙이지 말 것.
- 키는 두 개만 사용한다.
  - "requirements_markdown": 문자열 (Markdown 형식, 한국어)
  - "ascii_diagram": 문자열 (ASCII 다이어그램, 폭 80자 이내)

예시 (형식만 참고 – 실제 내용은 훨씬 길고 구체적으로 작성할 것):
<<JSON>>
{{
  "requirements_markdown": "# 개요\\n...긴 설명...\\n# 기능적 요구사항\\n...여러 항목...\\n# 비기능적 요구사항\\n...여러 항목...",
  "ascii_diagram": "[사용자] --> (로그인 화면)\\n(로그인 화면) --> (홈 화면)\\n..."
}}
</JSON>

[입력 정보]
- 앱/서비스 주제: "{goal}"
- 앱/서비스 분류: "{goal_topic}"

- 사용자가 선택한 주요 기능 목록 (체크박스):
{feat_json}

- 사용자가 선택한 대표적인 UI 컴포넌트:
{ui_json}

- 손그림에서 감지된 대략적인 화면 영역(있으면 참고, 없어도 무시 가능):
{comp_json}

{plan_section}

[작성 지침]
1) "requirements_markdown"
   - 한국어로 작성한다.
   - 전체 분량은 최소 500자 이상으로 충분히 자세히 쓴다.
   - 모든 단락과 목록은 Markdown의 불릿(`- `) 또는 번호 목록(`FR-01`, `NFR-01`) 형식으로 표기해 가독성을 높인다.
   - 구조 예:
     - # 개요
       - 서비스 목적, 주요 타깃 사용자, 해결하고자 하는 문제를 2~3문단으로 설명.
       - 각 문단은 `-` 불릿으로 시작하여 핵심 내용을 짧게 요약한다.
     - # 기능적 요구사항
       - FR-01, FR-02 처럼 번호를 붙여 나열한다.
       - 각 FR 항목 아래에는 2~3개의 하위 불릿을 사용해 입력/처리/출력, 예외 사항을 정리한다.
       - 최소 8개 이상의 기능적 요구사항을 작성한다.
       - 사용자가 체크한 기능 목록(feat_json)을 모두 반영하고,
         각 기능을 1개 이상의 요구사항으로 풀어서 쓴다.
       - 사용자가 선택한 UI 컴포넌트 목록(selected_ui)은 화면 설계의 힌트이므로, 요구사항 설명과 ASCII 다이어그램에 가능한 한 반영한다.
     - # 비기능적 요구사항
       - NFR-01, NFR-02 형식으로 번호를 붙인다.
       - 각 NFR에도 `-` 불릿을 활용해 측정 지표나 제약을 명시한다.
       - 최소 5개 이상의 비기능적 요구사항을 작성한다.
       - 성능(응답 시간, 동시 접속 수), 보안(인증, 인가, 데이터 보호),
         확장성, 사용성(UX), 로그/모니터링, 백엔드 API 설계 고려사항 등을 포함한다.

2) "ascii_diagram"
   - 전체적인 화면/기능 흐름을 화살표로 표현한다.
   - 괄호와 화살표를 사용해 이해하기 쉽게 표현한다.
   - 최소 5줄 이상 작성하며, 다음 요소를 포함한다.
     - [사용자]
     - (로그인 화면) 또는 (시작 화면)
     - (메인/홈 화면)
     - (주요 목록 화면) 예: (매칭 목록 화면), (콘텐츠 목록 화면) 등
     - (상세 화면)
   - 가능하면 백엔드 서버도 함께 표현한다. 예:
     [사용자] --> (로그인 화면)
     (로그인 화면) --> (백엔드 API 서버: 인증)
     (백엔드 API 서버: 인증) --> (홈 화면)

3) 반드시 아래 형식으로만 출력한다.
   - 맨 앞에 "<<JSON>>"
   - 맨 뒤에 "</JSON>"
   - 그 사이에는 JSON 한 개만 존재
   - 다른 텍스트는 절대 넣지 말 것.
   - 코드블록(````), 설명 문장, 중국어/영어 해설 등은 금지.
"""


def build_reasoning_prompt(
    goal: str,
    selected_features: List[Dict[str, str]],
    selected_ui: List[Dict[str, str]],
    detected_components: List[UIComponent],
) -> str:
    feat_text = json.dumps(selected_features, ensure_ascii=False)
    ui_text = json.dumps(selected_ui, ensure_ascii=False)
    comp_text = json.dumps([asdict(c) for c in detected_components], ensure_ascii=False)
    return f"""
너는 소프트웨어 요구사항을 분석하기 위한 선행 브레인스토밍 봇이다.

목표: 사용자가 제공한 입력을 검토하여 핵심 목표를 정제하고, 필요한 추가 정보가 있는지 판단한 뒤 reasoning 노트를 작성한다.

[입력]
- 원본 목표: "{goal}"
- 기능 선택: {feat_text}
- UI 컴포넌트 선택: {ui_text}
- 감지된 컴포넌트: {comp_text}

[출력 형식]
JSON만 출력하며, 다음 키를 포함한다:
{{
  "sanitized_goal": "한 문장으로 정제된 목표",
  "reasoning_summary": "Chain-of-Thought 형식의 요약 (3문장 이상)",
  "info_requests": ["추가로 필요한 정보 목록. 없으면 []"]
}}
"""


def build_plan_prompt(
    sanitized_goal: str,
    selected_features: List[Dict[str, str]],
    selected_ui: List[Dict[str, str]],
    reasoning_notes: str,
    observations: List[str],
) -> str:
    feat_text = json.dumps(selected_features, ensure_ascii=False)
    ui_text = json.dumps(selected_ui, ensure_ascii=False)
    obs_text = "\\n".join(observations[-3:]) if observations else "없음"
    return f"""
너는 SRS 생성을 위한 계획가다.

[입력 정보]
- 정제된 목표: "{sanitized_goal}"
- 선택 기능: {feat_text}
- 선택 UI: {ui_text}
- 최신 추론/관찰 노트: {obs_text}
- 참고 Reasoning: {reasoning_notes}

[출력 형식]
JSON만 출력한다:
{{
  "plan_title": "계획 이름",
  "plan_summary": "계획 요약 (3문장 이상)",
  "steps": [
    {{"id": "P1", "objective": "세부 목표", "actions": ["행동1", "행동2"], "expected_outputs": ["요구사항 섹션", "다이어그램 개선 포인트"]}}
  ]
}}
단, steps는 최소 3개 이상 작성한다.
"""


def build_plan_revision_prompt(
    sanitized_goal: str,
    previous_plan: str,
    errors: List[str],
    observations: List[str],
) -> str:
    obs_text = "\\n".join(observations[-5:]) if observations else "없음"
    err_text = "\\n".join(errors) if errors else "없음"
    return f"""
너는 실패한 계획을 개선하는 코치이다.

[목표]
- 정제된 목표: "{sanitized_goal}"
- 이전 계획(JSON): {previous_plan}
- 최근 관찰: {obs_text}
- 발생한 오류/검증 문제: {err_text}

[출력]
JSON만 반환하며, 다음 키를 포함한다:
{{
  "plan_title": "업데이트된 계획 이름",
  "plan_summary": "수정 요약 (2문단)",
  "steps": [
    {{"id": "R1", "objective": "개선 포인트", "actions": ["..."], "expected_outputs": ["..."]}}
  ]
}}
steps는 최소 2개 이상 작성하고, 이전 오류를 어떻게 다룰지 actions에 명시한다.
"""


# ------------------------------------------------------------------------------------
# JSON 추출 & 복구 로직 (Qwen이 이상하게 말해도 최대한 살려내기)
# ------------------------------------------------------------------------------------
def _find_first_valid_json(text: str) -> Dict[str, Any]:
    # 코드블록 제거
    text = text.replace("```json", "").replace("```", "").strip()

    # 모든 위치에서 { ... } 후보를 찾아보며 파싱 시도
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        for end_idx in range(len(text) - 1, start_idx, -1):
            if text[end_idx] != "}":
                continue
            candidate = text[start_idx : end_idx + 1]
            try:
                obj = json.loads(candidate)
                return obj
            except Exception:
                continue
    # 실패
    raise json.JSONDecodeError("No valid JSON object found", text, 0)


def _decode_jsonish_string(value: str) -> str:
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except Exception:
        return value.replace("\\n", "\n").replace("\\t", "\t")


def _recover_jsonish_fields(text: str) -> Optional[Dict[str, Any]]:
    def _extract(label: str) -> Optional[str]:
        token = f'"{label}"'
        start = text.find(token)
        if start == -1:
            return None
        colon = text.find(":", start + len(token))
        if colon == -1:
            return None
        quote = text.find('"', colon)
        if quote == -1:
            return None
        buf = []
        escaped = False
        i = quote + 1
        while i < len(text):
            ch = text[i]
            if escaped:
                buf.append(ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                break
            else:
                buf.append(ch)
            i += 1
        if not buf:
            return None
        return "".join(buf)

    rm_raw = _extract("requirements_markdown")
    ad_raw = _extract("ascii_diagram")
    recovered: Dict[str, Any] = {}
    if rm_raw:
        recovered["requirements_markdown"] = _decode_jsonish_string(rm_raw)
    if ad_raw:
        recovered["ascii_diagram"] = _decode_jsonish_string(ad_raw)
    return recovered or None


def extract_json(text: str) -> Dict[str, Any]:
    start_tag = "<<JSON>>"
    end_tag = "</JSON>"
    if start_tag in text and end_tag in text:
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag, start)
        text = text[start:end].strip()

    try:
        obj = _find_first_valid_json(text)
    except json.JSONDecodeError:
        recovered = _recover_jsonish_fields(text)
        if recovered:
            return recovered
        # 완전히 실패하면 통째로 requirements에 넣어둔다
        return {
            "requirements_markdown": text.replace("<<JSON>>", "").replace("</JSON>", "").strip(),
            "ascii_diagram": "(LLM JSON 파싱 실패: 위 텍스트를 수동으로 정리해야 함)",
        }

    # Qwen이 JSON 안에 JSON 문자열을 다시 넣는 경우 처리
    rm = obj.get("requirements_markdown")
    ad = obj.get("ascii_diagram")

    # requirements_markdown이 다시 JSON 문자열일 때
    if isinstance(rm, str) and rm.strip().startswith("{") and '"requirements_markdown"' in rm:
        try:
            inner = json.loads(rm)
            if "requirements_markdown" in inner:
                obj["requirements_markdown"] = inner["requirements_markdown"]
            if "ascii_diagram" in inner and not ad:
                obj["ascii_diagram"] = inner["ascii_diagram"]
        except Exception:
            pass

    # ascii_diagram이 JSON 문자열인 경우
    if isinstance(ad, str) and ad.strip().startswith("{") and '"ascii_diagram"' in ad:
        try:
            inner = json.loads(ad)
            if "ascii_diagram" in inner:
                obj["ascii_diagram"] = inner["ascii_diagram"]
        except Exception:
            pass

    # 최종 safety
    if "requirements_markdown" not in obj:
        obj["requirements_markdown"] = "(requirements_markdown 누락)"
    if "ascii_diagram" not in obj:
        obj["ascii_diagram"] = "(ascii_diagram 누락)"

    return obj


def safe_parse_json_block(text: str) -> Dict[str, Any]:
    try:
        return _find_first_valid_json(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


def format_plan_outline(plan_obj: Dict[str, Any]) -> str:
    try:
        return json.dumps(plan_obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(plan_obj)


def requirement_quality_tool(text: str) -> Dict[str, Any]:
    """간단한 품질 검사를 수행하는 내부 툴."""
    length = len(text or "")
    sections = [line for line in (text or "").splitlines() if line.startswith("#")]
    passed = length >= 500 and len(sections) >= 3
    return {
        "tool": "requirement_quality",
        "length": length,
        "section_count": len(sections),
        "passed": passed,
        "details": "분량>=500 & 섹션 3개 이상" if passed else "분량/섹션 기준 미달",
    }


def ascii_diagram_quality_tool(diagram: str) -> Dict[str, Any]:
    lines = [line for line in (diagram or "").splitlines() if line.strip()]
    includes_required = all(k in diagram for k in ["[사용자]", "(로그인", "(홈", "(상세"])
    passed = len(lines) >= 5 and includes_required
    return {
        "tool": "ascii_diagram_quality",
        "line_count": len(lines),
        "includes_required": includes_required,
        "passed": passed,
        "details": "라인>=5 & 필수 노드 포함" if passed else "다이어그램 기준 미달",
    }


def capture_gpu_metrics(stage: str) -> Dict[str, Any]:
    metric: Dict[str, Any] = {"stage": stage}
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        max_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        metric.update(
            {
                "device": f"cuda:{device}",
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "max_allocated_mb": round(max_alloc, 2),
            }
        )
    else:
        metric.update({"device": "cpu", "note": "CUDA unavailable"})
    return metric


def record_gpu_metric(state: RequirementState, stage: str) -> None:
    metric = capture_gpu_metrics(stage)
    state.setdefault("gpu_metrics", []).append(metric)


# ------------------------------------------------------------------------------------
# LangGraph 워크플로 구성
# ------------------------------------------------------------------------------------
def _run_llm(prompt: str) -> str:
    tokenizer, model = load_model()
    return model_generate(prompt, tokenizer, model)


def collect_inputs_node(state: RequirementState) -> RequirementState:
    """Streamlit 세션에서 받은 입력을 LangGraph 상태에 정리한다."""
    goal = state.get("goal", "").strip()
    state["goal"] = goal
    state["goal_topic"] = state.get("goal_topic") or classify_goal(goal)
    state.setdefault("selected_features", [])
    state.setdefault("selected_ui", [])
    state.setdefault("detected_components", [])
    state.setdefault("observations", [])
    state.setdefault("actions_taken", [])
    state["errors"] = []
    state.setdefault("tool_reports", [])
    state.setdefault("gpu_metrics", [])
    state["retry_count"] = state.get("retry_count", 0)
    return state


def reasoning_node(state: RequirementState) -> RequirementState:
    prompt = build_reasoning_prompt(
        goal=state.get("goal", ""),
        selected_features=state.get("selected_features", []),
        selected_ui=state.get("selected_ui", []),
        detected_components=state.get("detected_components", []),
    )
    record_gpu_metric(state, "reasoning_before")
    raw = _run_llm(prompt)
    record_gpu_metric(state, "reasoning_after")
    parsed = safe_parse_json_block(raw)
    sanitized_goal = parsed.get("sanitized_goal") or state.get("goal", "")
    state["sanitized_goal"] = sanitized_goal
    state["reasoning_notes"] = parsed.get("reasoning_summary", raw)
    state["info_requests"] = parsed.get("info_requests", [])
    state.setdefault("observations", []).append(f"Reasoning: {state['reasoning_notes']}")
    if state["info_requests"]:
        state["observations"].append(f"추가 정보 필요: {state['info_requests']}")
    state.setdefault("actions_taken", []).append("reasoning")
    return state


def plan_builder_node(state: RequirementState) -> RequirementState:
    prompt = build_plan_prompt(
        sanitized_goal=state.get("sanitized_goal", state.get("goal", "")),
        selected_features=state.get("selected_features", []),
        selected_ui=state.get("selected_ui", []),
        reasoning_notes=state.get("reasoning_notes", ""),
        observations=state.get("observations", []),
    )
    record_gpu_metric(state, "plan_builder_before")
    raw = _run_llm(prompt)
    record_gpu_metric(state, "plan_builder_after")
    plan_obj = safe_parse_json_block(raw)
    state["plan_outline"] = plan_obj
    state["plan_text"] = format_plan_outline(plan_obj)
    state.setdefault("actions_taken", []).append("plan_builder")
    return state


def plan_revision_node(state: RequirementState) -> RequirementState:
    prompt = build_plan_revision_prompt(
        sanitized_goal=state.get("sanitized_goal", state.get("goal", "")),
        previous_plan=state.get("plan_text", ""),
        errors=state.get("errors", []),
        observations=state.get("observations", []),
    )
    record_gpu_metric(state, "plan_revision_before")
    raw = _run_llm(prompt)
    record_gpu_metric(state, "plan_revision_after")
    plan_obj = safe_parse_json_block(raw)
    state["plan_outline"] = plan_obj
    state["plan_text"] = format_plan_outline(plan_obj)
    state.setdefault("actions_taken", []).append("plan_revision")
    state["errors"] = []
    return state


def action_prompt_builder_node(state: RequirementState) -> RequirementState:
    state["errors"] = []
    plan_text = state.get("plan_text", "")
    state["action_prompt"] = build_prompt(
        goal=state.get("goal", ""),
        goal_topic=state.get("goal_topic", "기타(범용)"),
        selected_features=state.get("selected_features", []),
        selected_ui=state.get("selected_ui", []),
        detected_components=state.get("detected_components", []),
        refined_goal=state.get("sanitized_goal"),
        plan_outline=plan_text,
    )
    state.setdefault("actions_taken", []).append("action_prompt_builder")
    return state


def llm_call_node(state: RequirementState) -> RequirementState:
    """로컬 Qwen 모델을 호출한다."""
    record_gpu_metric(state, "action_llm_before")
    state["llm_output"] = _run_llm(state.get("action_prompt", ""))
    record_gpu_metric(state, "action_llm_after")
    state.setdefault("actions_taken", []).append("action_llm")
    return state


def json_validator_node(state: RequirementState) -> RequirementState:
    """LLM 응답을 JSON으로 파싱하고 필수 키를 검증한다."""
    parsed = extract_json(state.get("llm_output", ""))
    state["parsed_json"] = parsed
    missing = []
    if not parsed.get("requirements_markdown"):
        missing.append("requirements_markdown")
    if not parsed.get("ascii_diagram"):
        missing.append("ascii_diagram")
    if missing:
        state.setdefault("errors", []).append(f"Missing keys: {missing}")
    state.setdefault("actions_taken", []).append("json_validator")
    return state


def tool_evaluation_node(state: RequirementState) -> RequirementState:
    parsed = state.get("parsed_json", {})
    req = parsed.get("requirements_markdown", "")
    diag = parsed.get("ascii_diagram", "")
    reports = [
        requirement_quality_tool(req),
        ascii_diagram_quality_tool(diag),
    ]
    state["tool_reports"] = reports
    failed = [r for r in reports if not r.get("passed")]
    if failed:
        state.setdefault("errors", []).append(f"Tool checks failed: {failed}")
    state.setdefault("observations", []).append(f"Tool reports: {reports}")
    state.setdefault("actions_taken", []).append("tool_evaluation")
    return state


def postprocess_node(state: RequirementState) -> RequirementState:
    """Streamlit UI에 보여줄 최종 데이터를 구성한다."""
    parsed = state.get("parsed_json", {})
    requirements = parsed.get("requirements_markdown", "(생성 실패)").strip()
    diagram = parsed.get("ascii_diagram", "(생성 실패)").strip()
    state["result_payload"] = {
        "requirements_markdown": requirements,
        "ascii_diagram": diagram,
        "debug_raw": parsed,
        "errors": state.get("errors", []),
        "plan": state.get("plan_outline", {}),
        "sanitized_goal": state.get("sanitized_goal", ""),
        "reasoning_notes": state.get("reasoning_notes", ""),
        "info_requests": state.get("info_requests", []),
        "actions_taken": state.get("actions_taken", []),
        "observations": state.get("observations", []),
        "tool_reports": state.get("tool_reports", []),
        "gpu_metrics": state.get("gpu_metrics", []),
    }
    state.setdefault("actions_taken", []).append("postprocess")
    return state


def observe_and_route_node(state: RequirementState) -> RequirementState:
    errors = state.get("errors", [])
    if errors:
        state.setdefault("observations", []).append(f"검증 오류: {errors}")
    else:
        state.setdefault("observations", []).append("검증 통과: 모든 필수 키 생성")
    retry_count = state.get("retry_count", 0)
    should_retry = bool(errors) and retry_count < MAX_AGENT_RETRIES
    state["should_retry"] = should_retry
    if should_retry:
        state["retry_count"] = retry_count + 1
    state.setdefault("actions_taken", []).append("observe")
    return state


def decide_next_step(state: RequirementState) -> str:
    return "retry" if state.get("should_retry") else "finish"


@st.cache_resource
def build_requirements_workflow():
    graph = StateGraph(RequirementState)
    graph.add_node("collect_inputs", collect_inputs_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("plan_builder", plan_builder_node)
    graph.add_node("action_prompt_builder", action_prompt_builder_node)
    graph.add_node("local_llm", llm_call_node)
    graph.add_node("json_validator", json_validator_node)
    graph.add_node("tool_evaluation", tool_evaluation_node)
    graph.add_node("postprocess", postprocess_node)
    graph.add_node("observe", observe_and_route_node)
    graph.add_node("plan_revision", plan_revision_node)

    graph.set_entry_point("collect_inputs")
    graph.add_edge("collect_inputs", "reasoning")
    graph.add_edge("reasoning", "plan_builder")
    graph.add_edge("plan_builder", "action_prompt_builder")
    graph.add_edge("plan_revision", "action_prompt_builder")
    graph.add_edge("action_prompt_builder", "local_llm")
    graph.add_edge("local_llm", "json_validator")
    graph.add_edge("json_validator", "tool_evaluation")
    graph.add_edge("tool_evaluation", "postprocess")
    graph.add_edge("postprocess", "observe")
    graph.add_conditional_edges(
        "observe",
        decide_next_step,
        {
            "retry": "plan_revision",
            "finish": END,
        },
    )
    return graph.compile()


# ------------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="SketchToSpec",
        page_icon="🖌️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 스타일 추가
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stHeader {
            font-family: 'Arial', sans-serif;
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Home 버튼 처리용 함수
    def reset_app():
        st.session_state.clear()
        st.rerun()

    if "home" not in st.session_state:
        st.session_state["home"] = True

    # ------------------------------
    # 홈 화면
    # ------------------------------
    if st.session_state["home"]:
        # Hero Section
        st.markdown(
            """
            <div style="text-align:center; padding-top:40px; padding-bottom:20px;">
                <h1 style="font-size:40px; margin-bottom:10px;">SketchToSpec</h1>
                <p style="font-size:18px; color:#555; margin-bottom:4px;">
                    손그림과 텍스트만으로 소프트웨어 요구사항 문서를 자동 생성하는 도구
                </p>
                <p style="font-size:15px; color:#888; margin-top:0;">
                    아이디어 · 기능 체크리스트 · 화면 스케치를 조합해
                    <br/>
                    백엔드 친화적인 요구사항 명세와 화면 흐름 다이어그램을 생성한다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # 5단계 Progress Guide 카드 UI
        st.markdown("### SketchToSpec 사용 흐름")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(
                "**1. 앱 주제 입력**\n"
                "만들고 싶은 앱의 목적과 타깃 사용자를 한 줄로 작성한다."
            )
        with col2:
            st.info(
                "**2. 기능 체크리스트 선택**\n"
                "로그인, 프로필, 목록, 검색 등 공통 기능을 체크박스로 선택한다."
            )
        with col3:
            st.info(
                "**3. UI 컴포넌트 선택**\n"
                "프로필 카드, 매칭 목록, 필터 바 같은 대표 화면 요소를 고른다."
            )

        col4, col5 = st.columns(2)
        with col4:
            st.info(
                "**4. 손그림 업로드 (선택)**\n"
                "종이에 그린 화면 스케치를 사진으로 업로드해 대략적인 레이아웃 힌트를 준다."
            )
        with col5:
            st.info(
                "**5. 요구사항 & ASCII 다이어그램 생성**\n"
                "LLM이 기능/화면 정보를 기반으로 요구사항 문서와 화면 흐름 다이어그램을 생성한다."
            )

        st.markdown("---")

        # 가운데 정렬된 시작 버튼
        spacer_left, center, spacer_right = st.columns([1, 1, 1])
        with center:
            if st.button("지금 시작하기", type="primary", use_container_width=True):
                st.session_state["home"] = False
                st.rerun()

    # ------------------------------
    # 실제 앱 기능 화면
    # ------------------------------
    else:
        st.header("앱 기능")
        if st.button("홈으로 돌아가기"):
            reset_app()

        # 세션 초기화
        st.session_state.setdefault("goal", "")
        st.session_state.setdefault("selected_features", [])  # 기능 체크리스트 선택 결과
        st.session_state.setdefault("selected_ui", [])        # UI 컴포넌트 선택 결과
        st.session_state.setdefault("image_bytes", None)
        st.session_state.setdefault("detected_components", [])
        st.session_state.setdefault("current_step", 1)        # 단계별 진행 상태

        # 상단 설명
        st.title("SketchToSpec")
        st.caption("앱 주제 + 자주 쓰는 기능 + 화면 요소 + 손그림을 바탕으로 요구사항 문서를 만들어주는 도구입니다.")

        st.markdown(
            """
    **사용 방법**

    1. 만들고 싶은 앱의 주제를 적습니다.  
    2. 이 앱에 들어갈 법한 "기능"을 체크박스로 고릅니다.  
    3. (선택) 주제에 맞는 UI 요소를 추천받고 선택합니다.  
    4. (선택) 손그림 스케치를 업로드합니다.  
    5. 요구사항 & 화면 흐름 다이어그램을 생성합니다.
            """
        )

        # ------------------------------
        # 단계별 UI 흐름
        # ------------------------------
        current_step = st.session_state.get("current_step", 1)

        if current_step == 1:
            st.subheader("1단계: 주제 & 기능 선택")
            goal = st.text_input(
                "어떤 앱을 만들고 싶나요?",
                value=st.session_state.get("goal", ""),
                placeholder="예) 대학생을 위한 소개팅 매칭 앱",
            )
            st.session_state["goal"] = goal.strip()

            st.markdown("---")
            st.subheader("이 앱에 필요한 기능들을 골라보세요")
            selected_features = []

            for category in sorted({f["category"] for f in FEATURE_LIBRARY}):
                with st.expander(f"카테고리: {category}", expanded=(category in ["인증/계정", "콘텐츠"])):
                    for feat in [f for f in FEATURE_LIBRARY if f["category"] == category]:
                        key = f"feat_{feat['key']}"
                        initial = any(sf["key"] == feat["key"] for sf in st.session_state.get("selected_features", []))
                        checked = st.checkbox(
                            f"{feat['name']} · {feat['desc']}",
                            key=key,
                            value=initial,
                        )
                        if checked:
                            selected_features.append(feat)

            st.session_state["selected_features"] = selected_features

            if st.button("다음 단계로"):
                st.session_state["current_step"] = 2
                st.rerun()

        elif current_step == 2:
            st.subheader("2단계: UI 컴포넌트 추천")

            goal = st.session_state.get("goal", "")
            if not goal:
                st.warning("1단계에서 앱 주제를 입력해 주세요.")
                if st.button("이전 단계로"):
                    st.session_state["current_step"] = 1
                    st.rerun()
            else:
                # 주제 분류 + 추천
                topic_label, recs = recommend_components(goal)
                st.session_state["goal_topic"] = topic_label  # 필요하면 이후 프롬프트에도 사용 가능

                # 분류 결과를 사용자에게 보여주기
                if topic_label == "기타(범용)":
                    st.info(
                        f"앱 주제를 별도 도메인으로 인식하지 못해 **{topic_label}**으로 분류했어요.\n"
                        f"아래는 다양한 서비스에 공통으로 쓸 수 있는 UI 컴포넌트 추천입니다."
                    )
                else:
                    st.info(
                        f"앱 주제를 **{topic_label}** 주제로 분류했고, "
                        f"이에 기반하여 아래 UI 컴포넌트를 추천합니다."
                    )

                st.write("이 앱에 들어갈 법한 화면 요소들을 골라보세요. (선택 사항)")

                selected_ui = []
                for comp in recs:
                    key = f"ui_{comp['name']}"
                    initial = any(c["name"] == comp["name"] for c in st.session_state.get("selected_ui", []))
                    checked = st.checkbox(
                        f"{comp['name']} · {comp['desc']}",
                        key=key,
                        value=initial,
                    )
                    if checked:
                        selected_ui.append(comp)

                st.session_state["selected_ui"] = selected_ui

                if st.button("다음 단계로"):
                    st.session_state["current_step"] = 3
                    st.rerun()
                if st.button("이전 단계로"):
                    st.session_state["current_step"] = 1
                    st.rerun()

        elif current_step == 3:
            st.subheader("3단계: 손그림 업로드")

            uploaded = st.file_uploader("PNG/JPG 파일 업로드", type=["png", "jpg", "jpeg"])
            if uploaded:
                image_bytes = uploaded.read()
                st.session_state["image_bytes"] = image_bytes

                st.image(image_bytes, caption="업로드된 손그림", use_container_width=True)

                comps = detect_components(image_bytes)
                st.session_state["detected_components"] = comps

                with st.expander("감지된 화면 영역 (참고용)", expanded=False):
                    st.json([asdict(c) for c in comps])
            else:
                st.info("손그림을 업로드하지 않으면, 기능/화면 요소만 기준으로 요구사항을 생성합니다.")

            if st.button("다음 단계로"):
                st.session_state["current_step"] = 4
                st.rerun()
            if st.button("이전 단계로"):
                st.session_state["current_step"] = 2
                st.rerun()

        elif current_step == 4:
            st.subheader("4단계: 요구사항 & 화면 흐름 생성")

            goal = st.session_state.get("goal", "").strip()
            selected_features = st.session_state.get("selected_features", [])
            selected_ui = st.session_state.get("selected_ui", [])
            detected_components = st.session_state.get("detected_components", [])
            has_img = st.session_state.get("image_bytes") is not None

            goal_topic = st.session_state.get("goal_topic", "기타(범용)")

            st.markdown("### 요약")
            st.write(f"**앱 주제**: {goal if goal else '입력되지 않음'}")
            st.write(f"**분류된 주제**: {goal_topic}")
            st.write(f"**선택된 기능**: {', '.join([f['name'] for f in selected_features]) if selected_features else '없음'}")
            st.write(f"**선택된 UI 컴포넌트**: {', '.join([ui['name'] for ui in selected_ui]) if selected_ui else '없음'}")
            st.write(f"**손그림 업로드 여부**: {'예' if has_img else '아니오'}")

            if not goal:
                st.warning("앱 주제가 비어 있습니다. 1단계로 돌아가 주제를 입력해 주세요.")
                if st.button("이전 단계로"):
                    st.session_state["current_step"] = 3
                    st.rerun()
                return

            workflow = build_requirements_workflow()

            if st.button("요구사항 문서 만들기", type="primary"):
                with st.spinner("LLM이 내용을 정리하는 중입니다..."):
                    try:
                        final_state = workflow.invoke(
                            {
                                "goal": goal,
                                "goal_topic": goal_topic,
                                "selected_features": selected_features,
                                "selected_ui": selected_ui,
                                "detected_components": detected_components,
                            }
                        )
                        payload = final_state["result_payload"]
                    except Exception as e:
                        st.error(f"생성 중 오류가 발생했습니다: {e}")
                        return

                st.markdown("---")
                st.markdown("### 📄 요구사항 (Markdown)")
                req_md = payload.get("requirements_markdown", "").strip()
                if req_md:
                    st.markdown(req_md)
                else:
                    st.info("요구사항 텍스트가 비어 있습니다.")

                st.markdown("### 📐 화면 흐름 (ASCII 다이어그램)")
                ascii_diag = payload.get("ascii_diagram", "").strip()
                if ascii_diag:
                    st.code(ascii_diag, language="text")
                else:
                    st.info("ASCII 다이어그램이 비어 있습니다.")

                with st.expander("LLM 원본 JSON (디버깅용)", expanded=False):
                    st.json(payload.get("debug_raw", {}))

                if payload.get("errors"):
                    st.warning("\n".join(payload["errors"]))

                with st.expander("에이전트 계획 & 관찰 로그", expanded=False):
                    st.markdown("**Plan Outline**")
                    st.json(payload.get("plan", {}))
                    st.markdown("**Observations**")
                    st.write(payload.get("observations", []))
                    st.markdown("**Actions Taken**")
                    st.write(payload.get("actions_taken", []))
                    st.markdown("**Tool Reports**")
                    st.json(payload.get("tool_reports", []))
                    st.markdown("**GPU Metrics**")
                    st.json(payload.get("gpu_metrics", []))
                    st.markdown("**Reasoning Trace**")
                    reasoning_trace = {
                        "sanitized_goal": payload.get("sanitized_goal"),
                        "reasoning_notes": payload.get("reasoning_notes"),
                        "info_requests": payload.get("info_requests"),
                    }
                    st.json(reasoning_trace)

            if st.button("이전 단계로"):
                st.session_state["current_step"] = 3
                st.rerun()


if __name__ == "__main__":
    main()
