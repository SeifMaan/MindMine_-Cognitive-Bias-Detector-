"""
Cognitive Bias Detector — Streamlit app (Mistral version)
---------------------------------------------------------
Pipeline:
  1. Fine-tuned DistilBERT → detects cognitive biases
  2. Mistral LLM → generates explanation + reframe
"""

import os
import json
import requests
import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cognitive Bias Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load Secrets (Hugging Face Spaces compatible) ────────────────────────────
# For Hugging Face Spaces, secrets are in os.environ
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Also check Streamlit secrets as fallback
if not MISTRAL_API_KEY:
    try:
        MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")
    except:
        pass

if not MISTRAL_API_KEY:
    st.error("⚠️ MISTRAL_API_KEY not found. Add it in Hugging Face Space Settings → Repository Secrets")
    st.stop()

# ── Load Classifier ───────────────────────────────────────────────────────────
MODEL_DIR = "./bias_model"


@st.cache_resource
def load_classifier():
    with open(f"{MODEL_DIR}/label_map.json") as f:
        label_map = json.load(f)

    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    return tokenizer, model, id2label


tokenizer, classifier, id2label = load_classifier()


# ── Bias Detection ────────────────────────────────────────────────────────────
def detect_biases(text: str, top_k=3, threshold=0.08):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = classifier(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    scored = [(id2label[i], float(p)) for i, p in enumerate(probs)]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [(b, c) for b, c in scored[:top_k] if c >= threshold]


# ── Mistral Explanation Generator ─────────────────────────────────────────────
def get_explanation(text: str, bias_name: str, tone: str):

    tone_map = {
        "Supportive": "warm, empathetic, encouraging",
        "Analytical": "logical, structured, precise",
        "Neutral": "balanced and objective",
    }

    prompt = f"""You are a cognitive bias expert.

TEXT:
{text}

BIAS:
{bias_name}

TONE:
{tone_map[tone]}

Return ONLY valid JSON:
{{
  "explanation": "2-3 sentences explaining the bias in this text",
  "reframe": "1-3 sentences offering a healthier perspective"
}}"""

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            raw = response.json()["choices"][0]["message"]["content"].strip()
            
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            
            return json.loads(raw)
        else:
            return {
                "explanation": f"This reflects patterns consistent with {bias_name}.",
                "reframe": "Try to consider alternative explanations and broader context.",
            }
            
    except Exception:
        return {
            "explanation": f"This reflects patterns consistent with {bias_name}.",
            "reframe": "Try to consider alternative explanations and broader context.",
        }


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🧠 Cognitive Bias Detector</div>', unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Paste text to analyze cognitive biases and thinking patterns.</div>',
    unsafe_allow_html=True,
)

text_input = st.text_area("Your text", height=150, label_visibility="collapsed")

col1, col2 = st.columns([2, 1])

with col1:
    tone = st.radio(
        "Explanation tone", ["Supportive", "Analytical", "Neutral"], horizontal=True
    )

with col2:
    top_k = st.slider("Max biases", 1, 3, 2)

analyze_btn = st.button("Analyze", use_container_width=True)


# ── Analysis ─────────────────────────────────────────────────────────────────
if analyze_btn:

    if not text_input.strip():
        st.warning("Enter some text first.")
        st.stop()

    with st.spinner("Detecting biases..."):
        detected = detect_biases(text_input, top_k=top_k)

    if not detected:
        st.success("No strong cognitive biases detected.")
        st.stop()

    for bias, conf in detected:

        with st.spinner(f"Analyzing {bias}..."):
            result = get_explanation(text_input, bias, tone)

        pct = int(conf * 100)

        st.markdown(
            f"""
        <div class="bias-card">
            <div class="bias-name">{bias}</div>
            <div class="confidence-label">Confidence</div>
            <div style="background:#eee; height:8px; border-radius:999px;">
                <div style="width:{pct}%; background:#6366f1; height:8px; border-radius:999px;"></div>
            </div>

            <div class="section-label">Why it appears</div>
            <div class="explanation-text">{result["explanation"]}</div>

            <div class="section-label">Reframe</div>
            <div class="reframe-box">{result["reframe"]}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Cognitive Bias Detector · Built by Seif</div>',
    unsafe_allow_html=True,
)
