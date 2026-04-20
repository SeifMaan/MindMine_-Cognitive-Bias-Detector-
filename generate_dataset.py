"""
generate_dataset.py
-------------------
Generates a synthetic training dataset for the Cognitive Bias Detector.
Uses Gemini Flash (free tier) to generate 60 examples per bias × 15 biases = 900 rows.
Output: biases_dataset.csv with columns: text, label

Usage:
    pip install google-generativeai pandas
    export GEMINI_API_KEY="your_key_here"
    python generate_dataset.py
"""

import os
import time
import json
import pandas as pd
from mistralai import Mistral

from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
EXAMPLES_PER_BIAS = 60
OUTPUT_FILE = "biases_dataset.csv"
RETRY_DELAY = 5  # seconds between retries on rate limit
MAX_RETRIES = 3

client = Mistral(api_key=MISTRAL_API_KEY)

MODEL_NAME = "mistral-large-latest"

# ── 15 Bias Definitions ──────────────────────────────────────────────────────
BIASES = {
    "Confirmation Bias": (
        "Seeking or interpreting information in a way that confirms pre-existing beliefs, "
        "ignoring contradictory evidence. "
        "Example: 'Every article I read proves that vaccines are dangerous — the evidence is overwhelming.'"
    ),
    "Black-or-White Thinking": (
        "Viewing situations in extremes with no middle ground, using absolute terms like always/never/everyone/no one. "
        "Example: 'If I'm not perfect at this, I'm a complete failure.'"
    ),
    "Overgeneralization": (
        "Drawing a broad conclusion from a single event or a small number of experiences. "
        "Example: 'I failed one exam, so I'm just bad at studying in general.'"
    ),
    "Catastrophizing": (
        "Expecting or imagining the worst possible outcome; blowing things out of proportion. "
        "Example: 'I made a typo in that email — my boss is going to think I'm incompetent and fire me.'"
    ),
    "Emotional Reasoning": (
        "Assuming something is true because it feels true, letting emotions override facts. "
        "Example: 'I feel like nobody likes me, so it must be true that I have no real friends.'"
    ),
    "Should Statements": (
        "Using rigid rules about how you or others must behave, often causing guilt or frustration. "
        "Example: 'I should always be productive. I shouldn't need rest — that's just laziness.'"
    ),
    "Labeling": (
        "Attaching a fixed, global label to yourself or others based on a specific event. "
        "Example: 'I burned dinner once — I'm just a terrible cook. He forgot my birthday, he's so selfish.'"
    ),
    "Jumping to Conclusions": (
        "Reaching a negative conclusion without sufficient evidence, assuming the worst without facts. "
        "Example: 'She didn't reply to my message — she must be angry at me.'"
    ),
    "Sunk Cost Fallacy": (
        "Continuing a behavior because of previously invested resources (time, money, effort) rather than future value. "
        "Example: 'I've already spent three years in this career — I can't quit now even though I'm miserable.'"
    ),
    "Anchoring Bias": (
        "Relying too heavily on the first piece of information encountered when making decisions. "
        "Example: 'The first apartment I saw was $2000/month, so $1800 feels like a great deal even though it's still too expensive.'"
    ),
    "Bandwagon Effect": (
        "Adopting beliefs or behaviors because many other people do, valuing popularity over evidence. "
        "Example: 'Everyone in my office is investing in crypto, so it must be a smart move.'"
    ),
    "Self-Serving Bias": (
        "Taking personal credit for successes while blaming external factors for failures. "
        "Example: 'I got the promotion because I'm talented. I got fired because my manager was unfair.'"
    ),
    "Optimism Bias": (
        "Overestimating the likelihood of positive outcomes and underestimating personal risk or negative outcomes. "
        "Example: 'I don't need to study much — I tend to do well in exams without much prep.'"
    ),
    "Personalization": (
        "Blaming yourself excessively for external events that are outside your control. "
        "Example: 'My friend is in a bad mood today — it must be something I said or did.'"
    ),
    "Availability Heuristic": (
        "Overestimating the likelihood of events based on how easily examples come to mind, often due to recent exposure. "
        "Example: 'I just read about a plane crash, so flying must be extremely dangerous — I'll drive instead.'"
    ),
}


# ── Prompt Builder ───────────────────────────────────────────────────────────
def build_prompt(bias_name: str, bias_description: str, n: int) -> str:
    return f"""You are generating a training dataset for a cognitive bias text classifier.

Bias to generate examples for: {bias_name}
Definition: {bias_description}

Generate exactly {n} realistic, diverse example sentences that clearly demonstrate this bias.
Requirements:
- Each sentence should be 1-3 sentences long (realistic human writing, not academic)
- Vary the context: work, relationships, health, money, self-image, social situations
- Vary the tone: casual, worried, confident, frustrated, resigned
- Do NOT include the bias name in the text
- Do NOT repeat the same structure
- Make them sound like real things a person would think or say

Return ONLY a valid JSON array of strings. No preamble, no explanation, no markdown fences.
Example format: ["sentence one", "sentence two", ...]
"""


# ── Generator ────────────────────────────────────────────────────────────────


def generate_examples(bias_name: str, bias_description: str, n: int) -> list[str]:
    prompt = build_prompt(bias_name, bias_description, n)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                # optional but helpful if supported
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content

            if not raw:
                raise ValueError("Empty response from API")

            raw = raw.strip()

            # 🧼 Clean common formatting issues (e.g. ```json blocks)
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # 🔍 Parse JSON
            parsed = json.loads(raw)

            # Handle both formats:
            # 1) {"examples": [...]}
            # 2) [...]
            if isinstance(parsed, dict):
                examples = parsed.get("examples", [])
            elif isinstance(parsed, list):
                examples = parsed
            else:
                raise ValueError("Unexpected JSON structure")

            if not isinstance(examples, list):
                raise ValueError("Response is not a list")

            # ✂️ Clean + normalize
            examples = [str(e).strip() for e in examples if str(e).strip()]

            if len(examples) < n:
                print(f"  ⚠️ Got {len(examples)}/{n} — retrying...")
                time.sleep(RETRY_DELAY)
                continue

            return examples[:n]

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ❌ Parse error on attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"  ❌ API error on attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))

    print(
        f"  ⛔ Failed to generate examples for '{bias_name}' after {MAX_RETRIES} attempts"
    )
    return []


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    all_rows = []

    print(
        f"🧠 Generating {EXAMPLES_PER_BIAS} examples × {len(BIASES)} biases = {EXAMPLES_PER_BIAS * len(BIASES)} total rows\n"
    )

    for i, (bias_name, bias_description) in enumerate(BIASES.items(), 1):
        print(f"[{i:02d}/{len(BIASES)}] Generating: {bias_name} ...")

        examples = generate_examples(bias_name, bias_description, EXAMPLES_PER_BIAS)

        for text in examples:
            all_rows.append({"text": text, "label": bias_name})

        print(f"  ✅ {len(examples)} examples collected")

        # Be polite to the free tier rate limits
        time.sleep(2)

    df = pd.DataFrame(all_rows)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Done! {len(df)} rows saved to '{OUTPUT_FILE}'")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
