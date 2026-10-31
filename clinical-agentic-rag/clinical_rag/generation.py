"""Step 5 - robust generation with direct API timeout handling to prevent socket crashes."""
import requests
import streamlit as st
from . import config

def generate_answer(query: str, agent_results: dict) -> str:
    if not query or not query.strip():
        return "Please enter a valid clinical research question."
    
    # Build context string safely
    evidence_texts = []
    for source, results in agent_results.items():
        for r in results or []:
            evidence_texts.append(f"[{source}]: {r.get('document', '')}")
    
    context = "\n\n".join(evidence_texts) if evidence_texts else "No relevant evidence found."
    
    prompt = f"""You are a clinical research assistant. Answer the user's question using ONLY the provided evidence. Do not invent facts.

QUESTION: {query}

RETRIEVED EVIDENCE:
{context}

ANSWER:"""

    try:
        api_url = f"https://api-inference.huggingface.co/models/{config.LLM_REPO_ID}"
        headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 512, "temperature": 0.1, "return_full_text": False},
            "options": {"wait_for_model": True}
        }
        
        # 25-second explicit timeout prevents hanging socket and Errno 104 connection resets
        response = requests.post(api_url, headers=headers, json=payload, timeout=25)
        
        if response.status_code == 200:
            res = response.json()
            if isinstance(res, list) and len(res) > 0:
                return res[0].get("generated_text", "").strip()
            elif isinstance(res, dict) and "generated_text" in res:
                return res["generated_text"].strip()
            return str(res)
        elif response.status_code == 503:
            return "⏳ **Model is warming up:** Hugging Face is loading the model into memory. Please wait 15 seconds and send your query again."
        elif response.status_code == 402:
            return "❌ **API Limit Reached:** Your Hugging Face account requires payment or has hit rate limits for this model endpoint."
        else:
            return f"❌ API Error ({response.status_code}): {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return "⏱️ **Request Timeout:** The Hugging Face server took too long to respond. Please try your query again."
    except Exception as e:
        return f"❌ Generation error: {str(e)}"
