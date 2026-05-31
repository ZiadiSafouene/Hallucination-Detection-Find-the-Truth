from __future__ import annotations

from .data import Example

SYSTEM_PROMPT = """أنت مقيّم دقيق باللغة العربية. مهمتك كشف هلوسة الإجابات المولدة واختيار الخيار الصحيح عندما يكون ذلك مطلوبًا."""


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{key}) {value}" for key, value in options.items())


def build_prompt(example: Example) -> str:
    options_text = format_options(example.options) if example.options else ""
    return f"""{SYSTEM_PROMPT}

السؤال:
{example.question}

الإجابة المولدة:
{example.generated_answer}

الإجابة المرجعية الصحيحة:
{example.gold_answer}

الخيارات:
{options_text}

أجب بصيغة JSON فقط، بدون أي تعليق أو شرح إضافي.

يجب أن تختار الخيار الصحيح من `options` في جميع الحالات، سواء كانت الإجابة المولدة صحيحة أو تحتوي على هلوسة.

إذا كانت الإجابة المولدة موثوقة وصحيحة، أجب كما يلي:
{{"predicted_label": "no_hallucination", "selected_option": "A"}}

إذا كانت الإجابة المولدة تحتوي على هلوسة أو معلومات غير مدعومة، أجب كما يلي:
{{"predicted_label": "hallucination", "selected_option": "A"}}

- استخدم فقط واحدة من القيم التالية للحقل `predicted_label`:
  - no_hallucination
  - hallucination
- استخدم مفتاحًا واحدًا فقط من الخانات الموجودة في `options` لحقل `selected_option`.
- لا تجعل `selected_option` فارغًا أو null.

JSON:"""