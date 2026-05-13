import gc
import html
import json
import os
import random
import re
import threading
import time
import traceback
import uuid
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "LoML.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo" / "outputs"
DEFAULT_DETECT_MODEL = PROJECT_ROOT / "model" / "detect" / "qwen_dpo"
DEFAULT_RETRIEVAL_MODEL = PROJECT_ROOT / "model" / "retrieval" / "qwen3-0.6b-finetuned-new" / "final" / "final"
DEFAULT_FAISS_INDEX = PROJECT_ROOT / "model" / "retrieval" / "index" / "udKB_index_0.6b.faiss"
DEFAULT_IDIOM_DICT = PROJECT_ROOT / "data" / "udKB.json"
DEFAULT_TRANSLATE_MODEL = PROJECT_ROOT / "model" / "base" / "Qwen3-8B"

MODEL_SIZE = "8B"
PARALLEL_SIZE = 1
DEFAULT_MAX_ITEMS = 5
DEFAULT_PREVIEW_ITEMS = 3

STAGE_TOTAL = 3
JOBS = {}
JOBS_LOCK = threading.Lock()


def log(message):
    print(f"[demo] {message}", flush=True)


def initial_job_state():
    return {
        "status": "queued",
        "stage_index": 0,
        "stage_total": STAGE_TOTAL,
        "stage_name": "Waiting",
        "message": "Waiting to start.",
        "model_percent": 0,
        "model_label": "Not started",
        "data_percent": 0,
        "data_label": "Not started",
        "logs": [],
        "result_html": "",
        "error_html": "",
        "updated_at": time.time(),
    }


def update_job(job_id, **updates):
    if not job_id:
        return
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        message = updates.get("message")
        if message:
            logs = job.setdefault("logs", [])
            if not logs or logs[-1] != message:
                logs.append(message)
                del logs[:-12]
        job.update(updates)
        job["updated_at"] = time.time()


def get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return deepcopy(job) if job else None


def make_progress(job_id):
    def progress(**updates):
        update_job(job_id, **updates)

    return progress


def emit(progress, message, **updates):
    log(message)
    if progress:
        progress(message=message, **updates)


DETECT_SYSTEM_TEMPLATE = """
# 核心方法

你将扮演一名严谨的语言分析师。你的核心方法论是：通过“字面含义”与“语境推断含义”的逻辑对比，来客观判断一个表达是否为“非常规表达”。

# 关键定义：非常规表达 (Key Definition: Unconventional Expression)

“非常规表达” (Unconventional Expression) 指的是一个词汇或短语，其整体含义不能从其组成单词的字面意思直接推断得出。这种表达的意义通常是比喻性的、惯用的、或具有特定文化背景的，并且高度依赖于上下文。它主要包括习语 (idioms)、俚语 (slang) 和隐喻 (metaphors) 等。

# 具体要求
1. 语句中的非常规表达可能包含人称、时态、缩写等变化，请从中识别出原始形态。
2. 常规用法中的非常规表达也纳入考虑范围。
3. 字面含义和比喻含义都可解释时，应结合语境选择最佳解释进行判断。
"""


DETECT_USER_TEMPLATE = """
包含非常规表达的示例句：
"When the share market crashed his fingers were burnt from all the investments that he had made ."

输出示例：

***Thought process:
“his fingers were burnt”：字面意思是“他的手指被烧伤”，但原句语境中表示因为投资不当而遭受损失，因此将其定义为非常规表达，供后续词典查阅。

***Output: ## YES ## 1.burn fingers


不包含非常规表达的示例句：
"She is reading a book."

输出示例：

***Thought process:
该句结构清晰，句中各个表达均为单词本身的字面意义，没有任何非常规表达。

***Output: ## NO


请根据系统提供的方法，并参考以上示例，处理以下英文句子中的非常规表达。
英文文本：{sentence}

请严格遵循以下输出格式：
***Thought process:
***Output: ## YES ## 1.<非常规表达>

或

***Thought process:
***Output: ## NO
"""


TRANSLATE_SYSTEM_TEMPLATE = """
You are a translation assistant whose task is to analyze translation background
information and ultimately provide the optimal Chinese translation.
You will be given:
1. The original English sentence.
2. The literal translation result from the model.
3. The detection result for unconventional expressions.
4. Dictionary entries retrieved by RAG.
"""


TRANSLATE_USER_TEMPLATE = """
Input Information:
1. Original English Sentence: {en}
2. Analysis Method and Detection Result for Unconventional Expressions in the Sentence: {think}
3. Retrieved Dictionary Entries: {rag_info}
Translation 0: Model's Literal Translation Result: {direct}

Output Format: Based on the retrieved dictionary entries, generate three separate translations:
Translation 1, Translation 2, and Translation 3.
Systematically analyze Translation 0, Translation 1, Translation 2, and Translation 3,
then determine which translation will be selected as the final output.

*** RAG Translation Results ***
Translation 1:
Translation 2:
Translation 3:

*** Translation Quality Analysis ***
Translation 0:
Translation 1:
Translation 2:
Translation 3:

*** Final Translation ***
(Output only the final translation result in this section; no other information is permitted)
"""


TRANSLATE_DIRECT_TEMPLATE = """
Translate the following English sentence into Chinese.
English: {sentence}
Chinese:
"""


TRANSLATE_FALLBACK_TEMPLATE = """
As a translation expert, please translate the sentence I provide to you into Chinese.
The sentence may contain unconventional expressions. Please first detect whether the sentence
contains any unconventional expressions. If they exist, provide an explanation before translating;
if they do not exist, translate the sentence directly.

Input sentence: {sentence}

Output Format:
**Whether it contains unconventional expressions (and their explanations)**
**Final Translation Result**
"""


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_path(raw_path):
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def display_path(raw_path):
    try:
        return os.path.relpath(Path(raw_path).resolve(), PROJECT_ROOT)
    except (OSError, ValueError):
        return Path(raw_path).name


def cleanup_cuda():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def build_chat_prompt(tokenizer, message):
    return tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def detect_step(data, model_path, progress=None):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    emit(
        progress,
        f"Stage 1/3 Detect: loading tokenizer from {display_path(model_path)}",
        stage_index=1,
        stage_name="Detect",
        model_percent=15,
        model_label="Loading tokenizer",
        data_percent=0,
        data_label="Waiting for model",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    emit(
        progress,
        "Stage 1/3 Detect: loading model weights",
        model_percent=45,
        model_label="Loading model",
    )
    sampling_params = SamplingParams(temperature=0, top_p=0.95, top_k=50, max_tokens=2048)
    llm = LLM(model=str(model_path), tensor_parallel_size=PARALLEL_SIZE)
    emit(progress, "Stage 1/3 Detect: model ready", model_percent=100, model_label="Model ready")

    emit(
        progress,
        f"Stage 1/3 Detect: building prompts for {len(data)} item(s)",
        data_percent=5,
        data_label="Building prompts",
    )
    prompts = []
    report_every = max(len(data) // 20, 1) if data else 1
    for item_index, entry in enumerate(data, start=1):
        sentence = entry.get("sentence", "")
        user_prompt = DETECT_USER_TEMPLATE.format(sentence=sentence)
        message = [
            {"role": "system", "content": DETECT_SYSTEM_TEMPLATE},
            {"role": "user", "content": user_prompt},
        ]
        prompts.append(build_chat_prompt(tokenizer, message))
        if item_index == len(data) or item_index % report_every == 0:
            percent = 5 + int(item_index / max(len(data), 1) * 25)
            progress(
                data_percent=percent,
                data_label=f"Building prompts {item_index}/{len(data)}",
            ) if progress else None

    emit(
        progress,
        "Stage 1/3 Detect: generating detection outputs",
        data_percent=60,
        data_label="Generating outputs",
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    assert len(data) == len(outputs), "detect outputs length does not match input data length"

    emit(progress, "Stage 1/3 Detect: parsing outputs", data_percent=85, data_label="Parsing outputs")
    report_every = max(len(data) // 20, 1) if data else 1
    for item_index, (entry, output) in enumerate(zip(data, outputs), start=1):
        text = output.outputs[0].text.strip()
        entry["detect_label"] = "Figurative" if "Output: ## YES" in text else "Without_Idiom"
        entry[f"dpo_eval_{MODEL_SIZE}"] = text
        if item_index == len(data) or item_index % report_every == 0:
            percent = 85 + int(item_index / max(len(data), 1) * 10)
            progress(
                data_percent=percent,
                data_label=f"Parsing outputs {item_index}/{len(data)}",
            ) if progress else None

    emit(progress, "Stage 1/3 Detect: releasing model", data_percent=98, data_label="Releasing model")
    del llm
    del tokenizer
    cleanup_cuda()
    emit(progress, "Stage 1/3 Detect: finished", data_percent=100, data_label="Detect complete")
    return data


def extract_outputs(text):
    """Extract detect_idiom from the raw detection model output."""
    if not isinstance(text, str):
        return []

    output_blocks = re.findall(
        r"\*\*\*Output:\s*##\s*YES\s*##\s*(.*?)(?=assistant|\n\s*\n|$)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not output_blocks:
        return []

    items = []
    for raw in output_blocks:
        raw = re.sub(r"assistant\s*$", "", raw.strip(), flags=re.IGNORECASE)
        if not raw:
            continue
        numbered_items = re.findall(r"\d+\.\s*(.*?)(?=\s*\d+\.|$)", raw)
        if numbered_items:
            items.extend(numbered_items)
        else:
            items.extend(re.split(r"[,;\n]+", raw))

    result = []
    seen = set()
    for item in items:
        cleaned = item.strip(" .-\t\r\n'\"")
        key = cleaned.lower()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def clean_detect_step(data, progress=None):
    emit(progress, "Stage 1/3 Detect: cleaning detected idioms", data_percent=96, data_label="Cleaning idioms")
    for entry in data:
        entry["detect_idiom"] = extract_outputs(entry.get(f"dpo_eval_{MODEL_SIZE}", ""))
    detected_count = sum(1 for entry in data if entry.get("detect_idiom"))
    emit(
        progress,
        f"Stage 1/3 Detect: {detected_count}/{len(data)} item(s) contain detected idioms",
        data_percent=100,
        data_label="Detect complete",
    )
    return data


def get_idiom_info(idiom_dict, idiom_id):
    idiom_id = int(idiom_id)
    if isinstance(idiom_dict, list):
        return idiom_dict[idiom_id]
    return idiom_dict[str(idiom_id)]


def retrieval_step(data, model_path, index_path, idiom_dict_path, progress=None):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    emit(
        progress,
        f"Stage 2/3 Retrieval: loading embedding model from {display_path(model_path)}",
        stage_index=2,
        stage_name="Retrieval",
        model_percent=20,
        model_label="Loading embedding model",
        data_percent=0,
        data_label="Waiting for retrieval assets",
    )
    model = SentenceTransformer(str(model_path))
    emit(
        progress,
        f"Stage 2/3 Retrieval: loading FAISS index {display_path(index_path)}",
        model_percent=60,
        model_label="Loading FAISS index",
    )
    index = faiss.read_index(str(index_path))
    emit(
        progress,
        f"Stage 2/3 Retrieval: loading idiom dictionary {display_path(idiom_dict_path)}",
        model_percent=85,
        model_label="Loading dictionary",
    )
    idiom_dict = load_json(idiom_dict_path)
    emit(progress, "Stage 2/3 Retrieval: retrieval assets ready", model_percent=100, model_label="Assets ready")

    total_queries = sum(len(entry.get("detect_idiom", [])) for entry in data)
    processed_queries = 0
    emit(
        progress,
        f"Stage 2/3 Retrieval: matching {total_queries} detected expression(s)",
        data_percent=0,
        data_label=f"0/{total_queries} expressions",
    )
    if total_queries == 0:
        emit(
            progress,
            "Stage 2/3 Retrieval: no detected expressions, adding empty retrieval results",
            data_percent=100,
            data_label="No detected expressions",
        )
    for entry_index, entry in enumerate(data, start=1):
        rag_info = []
        retrieval_debug = []
        for idiom in entry.get("detect_idiom", []):
            query_embedding = model.encode([idiom])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            distances, indices = index.search(query_embedding, 3)

            for idiom_id, score in zip(indices[0], distances[0]):
                matched_info = get_idiom_info(idiom_dict, idiom_id)
                rag_info.append(matched_info)
                retrieval_debug.append(
                    {
                        "query": idiom,
                        "matched_id": int(idiom_id),
                        "score": float(score),
                        "matched_info": matched_info,
                    }
                )
            processed_queries += 1
            if processed_queries % 5 == 0 or processed_queries == total_queries:
                percent = int(processed_queries / max(total_queries, 1) * 100)
                emit(
                    progress,
                    f"Stage 2/3 Retrieval: matched {processed_queries}/{total_queries} expression(s)",
                    data_percent=percent,
                    data_label=f"{processed_queries}/{total_queries} expressions",
                )

        entry["rag_info"] = rag_info
        entry["retrieval_top3"] = retrieval_debug

    del model
    cleanup_cuda()
    emit(progress, "Stage 2/3 Retrieval: finished", data_percent=100, data_label="Retrieval complete")
    return data


def translate_step(data, model_path, progress=None):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    emit(
        progress,
        f"Stage 3/3 Translate: loading tokenizer from {display_path(model_path)}",
        stage_index=3,
        stage_name="Translate",
        model_percent=15,
        model_label="Loading tokenizer",
        data_percent=0,
        data_label="Waiting for model",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    emit(
        progress,
        "Stage 3/3 Translate: loading translation model",
        model_percent=45,
        model_label="Loading model",
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, top_k=50, max_tokens=2048 * 4)
    llm = LLM(model=str(model_path), tensor_parallel_size=PARALLEL_SIZE)
    emit(progress, "Stage 3/3 Translate: model ready", model_percent=100, model_label="Model ready")

    emit(
        progress,
        f"Stage 3/3 Translate: building prompts for {len(data)} item(s)",
        data_percent=5,
        data_label="Building prompts",
    )
    prompts = []
    report_every = max(len(data) // 20, 1) if data else 1
    for item_index, entry in enumerate(data, start=1):
        if entry.get("detect_idiom", []) == []:
            user_prompt = TRANSLATE_DIRECT_TEMPLATE.format(sentence=entry.get("sentence", ""))
            message = [{"role": "user", "content": user_prompt}]
        elif entry.get("rag_info", []) != []:
            user_prompt = TRANSLATE_USER_TEMPLATE.format(
                en=entry.get("sentence", ""),
                direct=entry.get("direct_8b", entry.get("direct", "")),
                think=entry.get(f"dpo_eval_{MODEL_SIZE}", ""),
                rag_info=json.dumps(entry.get("rag_info", []), ensure_ascii=False),
            )
            message = [
                {"role": "system", "content": TRANSLATE_SYSTEM_TEMPLATE},
                {"role": "user", "content": user_prompt},
            ]
        else:
            user_prompt = TRANSLATE_FALLBACK_TEMPLATE.format(sentence=entry.get("sentence", ""))
            message = [{"role": "user", "content": user_prompt}]

        prompts.append(build_chat_prompt(tokenizer, message))
        if item_index == len(data) or item_index % report_every == 0:
            percent = 5 + int(item_index / max(len(data), 1) * 25)
            progress(
                data_percent=percent,
                data_label=f"Building prompts {item_index}/{len(data)}",
            ) if progress else None

    emit(
        progress,
        "Stage 3/3 Translate: generating translations",
        data_percent=60,
        data_label="Generating translations",
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    assert len(data) == len(outputs), "translate outputs length does not match input data length"

    emit(progress, "Stage 3/3 Translate: parsing translations", data_percent=85, data_label="Parsing outputs")
    report_every = max(len(data) // 20, 1) if data else 1
    for item_index, (entry, output) in enumerate(zip(data, outputs), start=1):
        entry["cot_result"] = output.outputs[0].text.strip()
        if item_index == len(data) or item_index % report_every == 0:
            percent = 85 + int(item_index / max(len(data), 1) * 10)
            progress(
                data_percent=percent,
                data_label=f"Parsing outputs {item_index}/{len(data)}",
            ) if progress else None

    emit(progress, "Stage 3/3 Translate: releasing model", data_percent=98, data_label="Releasing model")
    del llm
    del tokenizer
    cleanup_cuda()
    emit(progress, "Stage 3/3 Translate: finished", data_percent=100, data_label="Translate complete")
    return data


def choose_preview_indices(total_count, preview_count):
    if total_count <= 0 or preview_count <= 0:
        return []
    sample_size = min(total_count, preview_count)
    return sorted(random.sample(range(total_count), sample_size))


def preview(data, indices):
    return deepcopy([data[index] for index in indices if 0 <= index < len(data)])


def normalize_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def compact_value(value):
    if value is None or value == "":
        return "-"
    if isinstance(value, dict):
        preferred_keys = (
            "idiom",
            "definitions",
            "definition",
            "sense",
            "meaning",
            "description",
            "example",
            "sentence",
        )
        parts = []
        for key in preferred_keys:
            if key in value and value[key] not in (None, "", []):
                parts.append(f"{key}: {compact_value(value[key])}")
        if not parts:
            for key, item in list(value.items())[:4]:
                if item not in (None, "", []):
                    parts.append(f"{key}: {compact_value(item)}")
        return "; ".join(parts) if parts else "-"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return ", ".join(compact_value(item) for item in value)
    return str(value)


def render_field(label, value):
    return f"""
    <div class="field">
      <div class="field-label">{html.escape(label)}</div>
      <div class="field-value">{html.escape(compact_value(value))}</div>
    </div>
    """


def render_field_html(label, body_html):
    return f"""
    <div class="field">
      <div class="field-label">{html.escape(label)}</div>
      <div class="field-value">{body_html}</div>
    </div>
    """


def render_chips(values):
    values = [compact_value(value) for value in normalize_list(values) if compact_value(value) != "-"]
    if not values:
        return '<span class="empty-value">[]</span>'
    return '<div class="chips">' + "".join(
        f'<span class="chip">{html.escape(value)}</span>' for value in values
    ) + "</div>"


def render_record(index, inner_html):
    return f"""
    <article class="record">
      <div class="record-title">Item {index}</div>
      {inner_html}
    </article>
    """


def render_empty_stage():
    return '<p class="muted small">No preview items.</p>'


def render_input_preview(items):
    if not items:
        return render_empty_stage()
    return "".join(
        render_record(
            index,
            render_field("idiom", entry.get("idiom")) + render_field("sentence", entry.get("sentence")),
        )
        for index, entry in enumerate(items, start=1)
    )


def render_detect_preview(items):
    if not items:
        return render_empty_stage()
    return "".join(
        render_record(
            index,
            render_field("detect_label", entry.get("detect_label"))
            + render_field_html("detect_idiom", render_chips(entry.get("detect_idiom"))),
        )
        for index, entry in enumerate(items, start=1)
    )


def retrieval_groups(entry):
    idioms = [compact_value(item) for item in normalize_list(entry.get("detect_idiom")) if compact_value(item) != "-"]
    if not idioms:
        return []

    grouped_debug = {}
    for item in normalize_list(entry.get("retrieval_top3")):
        if not isinstance(item, dict):
            continue
        query = compact_value(item.get("query"))
        grouped_debug.setdefault(query, []).append(item)

    rag_info = normalize_list(entry.get("rag_info"))
    groups = []
    for index, idiom in enumerate(idioms):
        matches = grouped_debug.get(idiom, [])[:3]
        if not matches:
            fallback = rag_info[index * 3 : index * 3 + 3]
            matches = [{"matched_info": item} for item in fallback]
        groups.append({"query": idiom, "matches": matches})
    return groups


def get_match_title(info):
    if isinstance(info, dict):
        for key in ("idiom", "term", "phrase", "expression"):
            if info.get(key):
                return compact_value(info[key])
    return compact_value(info)


def get_match_definition(info):
    if isinstance(info, dict):
        for key in ("definitions", "definition", "sense", "meaning", "description"):
            if info.get(key):
                return compact_value(info[key])
    return ""


def render_match(match, index):
    info = match.get("matched_info", match) if isinstance(match, dict) else match
    title = get_match_title(info)
    definition = get_match_definition(info)
    meta = []
    if isinstance(match, dict) and match.get("matched_id") is not None:
        meta.append(f"ID {match['matched_id']}")
    if isinstance(match, dict) and match.get("score") is not None:
        meta.append(f"score {float(match['score']):.4f}")
    meta_html = f'<span class="match-meta">{html.escape(" | ".join(meta))}</span>' if meta else ""
    definition_html = (
        f'<div class="match-definition">{html.escape(definition)}</div>' if definition else ""
    )
    return f"""
    <li class="match">
      <div class="match-title"><span>{index}. {html.escape(title)}</span>{meta_html}</div>
      {definition_html}
    </li>
    """


def render_retrieval_preview(items):
    if not items:
        return render_empty_stage()

    blocks = []
    for index, entry in enumerate(items, start=1):
        groups = retrieval_groups(entry)
        if not groups:
            body = (
                render_field_html("detect_idiom", '<span class="empty-value">[]</span>')
                + render_field_html("top3 matches", '<span class="empty-value">[]</span>')
            )
        else:
            group_blocks = []
            for group in groups:
                matches = group["matches"][:3]
                if matches:
                    matches_html = "<ol>" + "".join(
                        render_match(match, rank) for rank, match in enumerate(matches, start=1)
                    ) + "</ol>"
                else:
                    matches_html = '<span class="empty-value">[]</span>'
                group_blocks.append(
                    f"""
                    <div class="query-block">
                      <div class="query-title">{html.escape(group["query"])}</div>
                      {matches_html}
                    </div>
                    """
                )
            body = "".join(group_blocks)
        blocks.append(render_record(index, body))
    return "".join(blocks)


def extract_final_translation(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    patterns = [
        r"\*{2,3}\s*Final Translation(?: Result)?\s*\*{2,3}\s*(?::|\uFF1A)?\s*(.*?)(?=\n\s*\*{2,3}\s*[^*\n]+?\s*\*{2,3}\s*|$)",
        r"Final Translation(?: Result)?\s*(?::|\uFF1A)\s*(.*?)(?=\n\s*\*{2,3}\s*[^*\n]+?\s*\*{2,3}\s*|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text


def translate_display_text(entry):
    cot_result = entry.get("cot_result", "")
    detected_idioms = [
        item for item in normalize_list(entry.get("detect_idiom"))
        if compact_value(item) != "-"
    ]
    if detected_idioms:
        return extract_final_translation(cot_result)
    return cot_result


def render_translate_preview(items):
    if not items:
        return render_empty_stage()
    return "".join(
        render_record(
            index,
            render_field("sentence", entry.get("sentence"))
            + render_field("final translation", translate_display_text(entry)),
        )
        for index, entry in enumerate(items, start=1)
    )


def render_stage_preview(stage, items):
    renderers = {
        "input": render_input_preview,
        "detect": render_detect_preview,
        "retrieval": render_retrieval_preview,
        "translate": render_translate_preview,
    }
    titles = {
        "input": "Input",
        "detect": "Detect",
        "retrieval": "Retrieval",
        "translate": "Translate",
    }
    renderer = renderers.get(stage)
    if renderer:
        body = renderer(items)
    else:
        body = "".join(render_record(index, render_field("item", item)) for index, item in enumerate(items, start=1))
    return f"""
    <section class="stage">
      <h2>{html.escape(titles.get(stage, stage))}</h2>
      {body}
    </section>
    """


def run_pipeline(form, job_id=None):
    progress = make_progress(job_id) if job_id else None
    if progress:
        progress(
            status="running",
            stage_index=0,
            stage_name="Input",
            message="Preparing pipeline inputs.",
            model_percent=0,
            model_label="Not started",
            data_percent=0,
            data_label="Validating paths",
        )

    input_path = resolve_path(form.get("input_path", [str(DEFAULT_INPUT)])[0])
    output_dir = resolve_path(form.get("output_dir", [str(DEFAULT_OUTPUT_DIR)])[0])
    detect_model = resolve_path(form.get("detect_model", [str(DEFAULT_DETECT_MODEL)])[0])
    retrieval_model = resolve_path(form.get("retrieval_model", [str(DEFAULT_RETRIEVAL_MODEL)])[0])
    faiss_index = resolve_path(form.get("faiss_index", [str(DEFAULT_FAISS_INDEX)])[0])
    idiom_dict = resolve_path(form.get("idiom_dict", [str(DEFAULT_IDIOM_DICT)])[0])
    translate_model = resolve_path(form.get("translate_model", [str(DEFAULT_TRANSLATE_MODEL)])[0])
    max_items = int(form.get("max_items", [str(DEFAULT_MAX_ITEMS)])[0] or 0)
    preview_items = int(form.get("preview_items", [str(DEFAULT_PREVIEW_ITEMS)])[0] or DEFAULT_PREVIEW_ITEMS)

    required_files = [input_path, faiss_index, idiom_dict]
    required_dirs = [detect_model, retrieval_model, translate_model]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
    for path in required_dirs:
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    emit(
        progress,
        f"Loading input data from {display_path(input_path)}",
        data_percent=20,
        data_label="Loading input JSON",
    )
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")
    if max_items > 0:
        data = data[:max_items]
    log(f"Loaded {len(data)} item(s) from {input_path}")
    preview_indices = choose_preview_indices(len(data), preview_items)
    log(f"Selected {len(preview_indices)} random preview item(s)")
    if progress:
        progress(
            message=f"Loaded {len(data)} item(s); selected {len(preview_indices)} preview item(s).",
            data_percent=100,
            data_label="Input ready",
        )

    paths = {
        "input": str(input_path),
        "detect": str(output_dir / "01_detect_result.json"),
        "retrieval": str(output_dir / "02_retrieval.json"),
        "translate": str(output_dir / "03_translate_result.json"),
    }

    result = {
        "paths": paths,
        "count": len(data),
        "preview_count": len(preview_indices),
        "previews": {},
    }
    result["previews"]["input"] = preview(data, preview_indices)

    emit(
        progress,
        "Stage 1/3 Detect: starting",
        stage_index=1,
        stage_name="Detect",
        model_percent=0,
        model_label="Starting",
        data_percent=0,
        data_label="Waiting",
    )
    detected = detect_step(deepcopy(data), detect_model, progress=progress)
    detected = clean_detect_step(detected, progress=progress)
    save_json(paths["detect"], detected)
    log(f"Saved detect result: {paths['detect']}")
    result["previews"]["detect"] = preview(detected, preview_indices)

    emit(
        progress,
        "Stage 2/3 Retrieval: starting",
        stage_index=2,
        stage_name="Retrieval",
        model_percent=0,
        model_label="Starting",
        data_percent=0,
        data_label="Waiting",
    )
    retrieved = retrieval_step(detected, retrieval_model, faiss_index, idiom_dict, progress=progress)
    save_json(paths["retrieval"], retrieved)
    log(f"Saved retrieval result: {paths['retrieval']}")
    result["previews"]["retrieval"] = preview(retrieved, preview_indices)

    emit(
        progress,
        "Stage 3/3 Translate: starting",
        stage_index=3,
        stage_name="Translate",
        model_percent=0,
        model_label="Starting",
        data_percent=0,
        data_label="Waiting",
    )
    translated = translate_step(retrieved, translate_model, progress=progress)
    save_json(paths["translate"], translated)
    log(f"Saved translate result: {paths['translate']}")
    result["previews"]["translate"] = preview(translated, preview_indices)

    emit(
        progress,
        "Pipeline finished.",
        stage_index=3,
        stage_name="Complete",
        model_percent=100,
        model_label="Complete",
        data_percent=100,
        data_label="Complete",
    )
    return result


def render_result_panel(result):
    path_items = "".join(
        f"<li><code>{html.escape(name)}</code>: {html.escape(display_path(path))}</li>"
        for name, path in result["paths"].items()
    )
    preview_blocks = [
        render_stage_preview(stage, items)
        for stage, items in result["previews"].items()
    ]
    return f"""
    <div class="result ok">
      <h2>Run Complete</h2>
      <p>Processed <strong>{result["count"]}</strong> item(s).</p>
      <h3>Saved Files</h3>
      <ul>{path_items}</ul>
      <h3>Preview</h3>
      <p class="muted small">Showing <strong>{result["preview_count"]}</strong> random concise preview item(s). Full outputs are saved in the files above.</p>
      {''.join(preview_blocks)}
    </div>
    """


def render_error_panel(error):
    return f"""
    <div class="result error">
      <h2>Run Failed</h2>
      <pre>{html.escape(error)}</pre>
    </div>
    """


def page_script():
    return r"""
<script>
(() => {
  const form = document.getElementById("run-form");
  const runButton = document.getElementById("run-button");
  const progressPanel = document.getElementById("progress-panel");
  const progressTitle = document.getElementById("progress-title");
  const stageBadge = document.getElementById("stage-badge");
  const resultSlot = document.getElementById("result-slot");
  const logList = document.getElementById("progress-log");

  function clampPercent(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return 0;
    }
    return Math.max(0, Math.min(100, numeric));
  }

  function setBar(prefix, percent, label, status) {
    const value = clampPercent(percent);
    const fill = document.getElementById(prefix + "-fill");
    const text = document.getElementById(prefix + "-text");
    fill.style.width = value + "%";
    fill.classList.toggle("active", status === "running" && value > 0 && value < 100);
    text.textContent = (label || "Waiting") + " | " + Math.round(value) + "%";
  }

  function updateSteps(state) {
    document.querySelectorAll(".progress-step").forEach((step) => {
      const index = Number(step.dataset.step);
      const active = state.status === "running" && state.stage_index === index;
      const done = state.status === "done" || state.stage_index > index;
      step.classList.toggle("active", active);
      step.classList.toggle("done", done);
    });
  }

  function updateLogs(logs) {
    logList.innerHTML = "";
    (logs || []).slice(-12).forEach((message) => {
      const item = document.createElement("li");
      item.textContent = message;
      logList.appendChild(item);
    });
  }

  function updateProgress(state) {
    progressPanel.classList.remove("hidden");
    const stageIndex = state.stage_index || 0;
    const total = state.stage_total || 3;
    const stageName = state.stage_name || "Preparing";
    const stageText = stageIndex ? "Stage " + stageIndex + "/" + total + " | " + stageName : stageName;
    progressTitle.textContent = state.message || stageText;
    stageBadge.textContent = state.status === "done" ? "Complete" : stageText;
    stageBadge.classList.toggle("done", state.status === "done");
    stageBadge.classList.toggle("error", state.status === "error");
    setBar("model", state.model_percent, state.model_label, state.status);
    setBar("data", state.data_percent, state.data_label, state.status);
    updateSteps(state);
    updateLogs(state.logs);
  }

  async function poll(jobId) {
    const response = await fetch("/status?job_id=" + encodeURIComponent(jobId), { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Unable to read job status.");
    }
    const state = await response.json();
    updateProgress(state);
    if (state.status === "done") {
      resultSlot.innerHTML = state.result_html || "";
      runButton.disabled = false;
      return;
    }
    if (state.status === "error") {
      resultSlot.innerHTML = state.error_html || "";
      runButton.disabled = false;
      return;
    }
    window.setTimeout(() => poll(jobId).catch(showClientError), 900);
  }

  function showClientError(error) {
    progressPanel.classList.remove("hidden");
    stageBadge.textContent = "Client Error";
    stageBadge.classList.add("error");
    progressTitle.textContent = error.message || "Client error.";
    runButton.disabled = false;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resultSlot.innerHTML = "";
    runButton.disabled = true;
    updateProgress({
      status: "running",
      stage_index: 0,
      stage_total: 3,
      stage_name: "Submitting",
      message: "Submitting pipeline job.",
      model_percent: 0,
      model_label: "Not started",
      data_percent: 0,
      data_label: "Submitting form",
      logs: ["Submitting pipeline job."]
    });

    try {
      const response = await fetch("/run", {
        method: "POST",
        body: new URLSearchParams(new FormData(form))
      });
      if (!response.ok) {
        throw new Error("Unable to start pipeline.");
      }
      const payload = await response.json();
      poll(payload.job_id).catch(showClientError);
    } catch (error) {
      showClientError(error);
    }
  });
})();
</script>
"""


def render_page(result=None, error=None):
    result_html = render_result_panel(result) if result else ""
    error_html = render_error_panel(error) if error else ""

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>DeReA Pipeline Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #172033;
      --muted: #5d6678;
      --accent: #1f7a68;
      --accent-dark: #155a4c;
      --accent-soft: #e7f4f1;
      --ok: #1f7a68;
      --shadow: 0 18px 45px rgba(23, 32, 51, 0.08);
      --danger-bg: #fff1f0;
      --danger-line: #ffccc7;
      --danger-text: #a8071a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #f2f6f5 0%, var(--bg) 260px);
      color: var(--text);
    }}
    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 28px auto 56px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p {{ color: var(--muted); line-height: 1.6; }}
    form, .result, .progress-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin-top: 18px;
      box-shadow: var(--shadow);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px 16px;
    }}
    label {{ display: grid; gap: 6px; font-size: 14px; font-weight: 700; }}
    input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
    }}
    input:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(31, 122, 104, 0.12);
      outline: none;
    }}
    .wide {{ grid-column: 1 / -1; }}
    button {{
      margin-top: 16px;
      border: 0;
      border-radius: 6px;
      padding: 11px 18px;
      font: inherit;
      font-weight: 700;
      color: white;
      background: var(--accent);
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-dark); }}
    button:disabled {{
      background: #9aaba7;
      cursor: progress;
    }}
    code {{
      background: #eef2f5;
      border-radius: 4px;
      padding: 2px 5px;
    }}
    .hidden {{ display: none !important; }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 13px; }}
    .eyebrow {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    .progress-head {{
      align-items: start;
      display: flex;
      gap: 16px;
      justify-content: space-between;
    }}
    .progress-head h2 {{
      font-size: 22px;
      margin: 4px 0 0;
    }}
    .stage-badge {{
      background: var(--accent-soft);
      border: 1px solid #bddbd4;
      border-radius: 999px;
      color: var(--accent-dark);
      flex: 0 0 auto;
      font-size: 13px;
      font-weight: 800;
      padding: 6px 10px;
    }}
    .stage-badge.done {{
      background: #edf8ef;
      border-color: #b7e0c0;
      color: var(--ok);
    }}
    .stage-badge.error {{
      background: var(--danger-bg);
      border-color: var(--danger-line);
      color: var(--danger-text);
    }}
    .stepper {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(3, 1fr);
      margin-top: 18px;
    }}
    .progress-step {{
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: flex;
      gap: 10px;
      padding: 10px;
    }}
    .progress-step span {{
      align-items: center;
      background: #eef2f5;
      border-radius: 50%;
      color: var(--muted);
      display: inline-flex;
      font-size: 13px;
      font-weight: 800;
      height: 26px;
      justify-content: center;
      width: 26px;
    }}
    .progress-step.active {{
      border-color: #98c8bd;
      background: #f5fbf9;
    }}
    .progress-step.active span,
    .progress-step.done span {{
      background: var(--accent);
      color: white;
    }}
    .progress-step.done {{
      border-color: #c6dfd9;
    }}
    .progress-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: 1fr 1fr;
      margin-top: 18px;
    }}
    .progress-row {{
      min-width: 0;
    }}
    .progress-label {{
      align-items: center;
      display: flex;
      gap: 12px;
      justify-content: space-between;
      margin-bottom: 8px;
    }}
    .progress-label span {{
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }}
    .bar {{
      background: #edf1f4;
      border-radius: 999px;
      height: 10px;
      overflow: hidden;
      position: relative;
    }}
    .bar-fill {{
      background: linear-gradient(90deg, var(--accent), #47a88e);
      border-radius: inherit;
      height: 100%;
      transition: width 260ms ease;
      width: 0;
    }}
    .bar-fill.active {{
      animation: pulse-fill 1.1s ease-in-out infinite alternate;
    }}
    .log-panel {{
      border-top: 1px solid var(--line);
      margin-top: 18px;
      padding-top: 12px;
    }}
    #progress-log {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      margin: 8px 0 0;
      padding-left: 18px;
    }}
    #progress-log li + li {{
      margin-top: 4px;
    }}
    @keyframes pulse-fill {{
      from {{ filter: brightness(1); }}
      to {{ filter: brightness(1.18); }}
    }}
    .stage {{
      border-top: 1px solid var(--line);
      margin-top: 20px;
      padding-top: 16px;
    }}
    .stage h2 {{
      margin: 0 0 12px;
      font-size: 20px;
    }}
    .record {{
      border-left: 3px solid var(--accent);
      padding: 8px 0 10px 14px;
      margin: 12px 0;
    }}
    .record-title {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
      text-transform: uppercase;
    }}
    .field {{
      display: grid;
      grid-template-columns: 140px minmax(0, 1fr);
      gap: 12px;
      padding: 5px 0;
    }}
    .field-label {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }}
    .field-value {{
      min-width: 0;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      border: 1px solid #c8d8d4;
      border-radius: 999px;
      background: #f1f8f6;
      color: var(--accent-dark);
      padding: 2px 8px;
      font-size: 13px;
      font-weight: 700;
    }}
    .empty-value {{
      color: var(--muted);
      font-family: Consolas, monospace;
    }}
    .query-block {{
      margin: 8px 0 14px;
    }}
    .query-title {{
      color: var(--accent-dark);
      font-weight: 700;
      margin-bottom: 6px;
    }}
    ol {{
      margin: 0;
      padding-left: 0;
      list-style: none;
    }}
    .match {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      margin: 8px 0;
      background: #fbfcfd;
    }}
    .match-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-weight: 700;
    }}
    .match-meta {{
      color: var(--muted);
      flex: 0 0 auto;
      font-size: 12px;
      font-weight: 400;
    }}
    .match-definition {{
      color: var(--muted);
      line-height: 1.55;
      margin-top: 4px;
    }}
    pre {{
      max-height: 420px;
      overflow: auto;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    section {{ margin-top: 18px; }}
    .error {{
      background: var(--danger-bg);
      border-color: var(--danger-line);
    }}
    @media (max-width: 760px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .wide {{ grid-column: auto; }}
      .progress-head {{ display: block; }}
      .stage-badge {{ display: inline-block; margin-top: 10px; }}
      .stepper {{ grid-template-columns: 1fr; }}
      .progress-grid {{ grid-template-columns: 1fr; }}
      .progress-label {{ display: block; }}
      .progress-label span {{ display: block; margin-top: 3px; text-align: left; }}
      .field {{ grid-template-columns: 1fr; gap: 2px; }}
      .match-title {{ display: block; }}
      .match-meta {{ display: block; margin-top: 3px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>DeReA Pipeline Demo</h1>
    <p>输入本地 JSON 文件路径后，依次执行 detect、retrieval、translate。每一步都会保存完整 JSON 文件，页面仅展示精简后的关键结果。</p>

    <form id="run-form" method="post" action="/run">
      <div class="grid">
        <label class="wide">Input JSON Path
          <input name="input_path" value="{html.escape(str(DEFAULT_INPUT))}">
        </label>
        <label class="wide">Output Directory
          <input name="output_dir" value="{html.escape(str(DEFAULT_OUTPUT_DIR))}">
        </label>
        <label>Detect Model
          <input name="detect_model" value="{html.escape(str(DEFAULT_DETECT_MODEL))}">
        </label>
        <label>Translate Model
          <input name="translate_model" value="{html.escape(str(DEFAULT_TRANSLATE_MODEL))}">
        </label>
        <label>Retrieval Model
          <input name="retrieval_model" value="{html.escape(str(DEFAULT_RETRIEVAL_MODEL))}">
        </label>
        <label>FAISS Index
          <input name="faiss_index" value="{html.escape(str(DEFAULT_FAISS_INDEX))}">
        </label>
        <label class="wide">Idiom Dictionary JSON
          <input name="idiom_dict" value="{html.escape(str(DEFAULT_IDIOM_DICT))}">
        </label>
        <label>Max Items
          <input name="max_items" type="number" min="0" value="{DEFAULT_MAX_ITEMS}">
        </label>
        <label>Preview Items
          <input name="preview_items" type="number" min="0" value="{DEFAULT_PREVIEW_ITEMS}">
        </label>
      </div>
      <button id="run-button" type="submit">Run Pipeline</button>
    </form>

    <section id="progress-panel" class="progress-panel hidden" aria-live="polite">
      <div class="progress-head">
        <div>
          <div class="eyebrow">Live Progress</div>
          <h2 id="progress-title">Waiting to start.</h2>
        </div>
        <span id="stage-badge" class="stage-badge">Idle</span>
      </div>

      <div class="stepper">
        <div class="progress-step" data-step="1"><span>1</span><strong>Detect</strong></div>
        <div class="progress-step" data-step="2"><span>2</span><strong>Retrieval</strong></div>
        <div class="progress-step" data-step="3"><span>3</span><strong>Translate</strong></div>
      </div>

      <div class="progress-grid">
        <div class="progress-row">
          <div class="progress-label">
            <strong>Model Loading</strong>
            <span id="model-text">Not started | 0%</span>
          </div>
          <div class="bar"><div id="model-fill" class="bar-fill"></div></div>
        </div>
        <div class="progress-row">
          <div class="progress-label">
            <strong>Data Processing</strong>
            <span id="data-text">Not started | 0%</span>
          </div>
          <div class="bar"><div id="data-fill" class="bar-fill"></div></div>
        </div>
      </div>

      <div class="log-panel">
        <div class="field-label">Recent Log</div>
        <ul id="progress-log"></ul>
      </div>
    </section>

    <div id="result-slot">
      {error_html}
      {result_html}
    </div>
  </main>
  {page_script()}
</body>
</html>"""


def run_job(job_id, form):
    try:
        result = run_pipeline(form, job_id=job_id)
        update_job(
            job_id,
            status="done",
            stage_index=3,
            stage_name="Complete",
            message="Pipeline complete.",
            model_percent=100,
            model_label="Complete",
            data_percent=100,
            data_label="Complete",
            result_html=render_result_panel(result),
        )
    except Exception:
        error = traceback.format_exc()
        update_job(
            job_id,
            status="error",
            message="Pipeline failed.",
            error_html=render_error_panel(error),
        )


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            query = parse_qs(parsed.query)
            job_id = query.get("job_id", [""])[0]
            job = get_job(job_id)
            if not job:
                self.respond_json({"status": "missing", "message": "Job not found."}, status=404)
                return
            self.respond_json(job)
            return
        self.respond(render_page())

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/run":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body)
        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = initial_job_state()
        thread = threading.Thread(target=run_job, args=(job_id, form), daemon=True)
        thread.start()
        self.respond_json({"job_id": job_id})

    def respond(self, body, status=200):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_json(self, payload, status=200):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    host = "127.0.0.1"
    port = 7860
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Demo running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
