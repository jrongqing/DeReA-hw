import argparse
import re
from pathlib import Path


from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


PARALLEL_SIZE = 1
DEFAULT_SENTENCE = "I do n't believe that he did n't take the money , but I will give him the benefit of the doubt until I can prove otherwise ."

model_path = "model/detect/qwen_dpo"
model_path = str(Path(__file__).resolve().parents[1] / model_path)
# model_path = "/data0/jrq/slang/DeReA-hw/model/detect/qwen_dpo"

tokenizer = AutoTokenizer.from_pretrained(model_path)
sampling_params = SamplingParams(temperature=0, top_p=0.95, top_k=50, max_tokens=2048)
llm = LLM(model=model_path, tensor_parallel_size=PARALLEL_SIZE)


system_template = """
You are a careful linguistic analyst.

Your task is to determine whether an English sentence contains an unconventional
expression. An unconventional expression is a word or phrase whose overall
meaning cannot be directly inferred from the literal meaning of its parts.
This includes idioms, slang, metaphors, and fixed figurative expressions.

Please compare the literal meaning with the contextual meaning and decide
objectively. If an expression has both literal and figurative readings, choose
the best reading according to the sentence context.
"""

user_prompt_template = """
Example with an unconventional expression:
"When the share market crashed his fingers were burnt from all the investments that he had made ."

***Thought process:
"his fingers were burnt" literally means his fingers were physically burned,
but in this context it means he suffered losses from bad investments.

***Output: ## YES ## 1.burn fingers


Example without an unconventional expression:
"She is reading a book."

***Thought process:
The sentence is literal. There is no hidden idiomatic or figurative meaning.

***Output: ## NO


Now analyze the following English sentence:
{sentence}

Please strictly use one of these output formats:

***Thought process:
***Output: ## YES ## 1.<unconventional expression>

or

***Thought process:
***Output: ## NO
"""


def build_prompt(sentence):
    user_prompt = user_prompt_template.format(sentence=sentence)
    message = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def detect_sentence(sentence):
    prompt = build_prompt(sentence)
    output = llm.generate([prompt], sampling_params=sampling_params)[0].outputs[0].text.strip()
    label = "Figurative" if "Output: ## YES" in output else "Without_Idiom"
    return label, output

def extract_outputs(text):
    """Extract phrases after each '***Output: ## YES ##' block."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Detect unconventional expressions in one English sentence."
    )
    parser.add_argument(
        "sentence",
        nargs="?",
        default=DEFAULT_SENTENCE,
        help="English sentence to analyze.",
    )
    args = parser.parse_args()

    label, output = detect_sentence(args.sentence)

    detect_result = extract_outputs(output)
    print("\nDetect result:")
    print(detect_result)
    ## ['give someone the benefit of the doubt']


if __name__ == "__main__":
    main()

