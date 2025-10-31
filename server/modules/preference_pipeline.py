import json
import requests

OLLAMA_URL = "http://ollama:11434/api/generate"


def query_ollama(model: str, text: str) -> str:
    """Query a local Ollama model with raw text."""
    payload = {
        "model": model,
        "prompt": text,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ Error querying Ollama ({model}):", e)
        return ""


def safe_json_parse(text: str, fallback):
    """Safely parse JSON output."""
    try:
        return json.loads(text.replace("json", "").replace("`", ""))
    except json.JSONDecodeError:
        print("⚠️ JSON parsing failed. Raw output:")
        print(text)
        return fallback


def run_pipeline(utterance: str):
    """Run the full 3-step pipeline in sequence."""
    print(f"\n🎯 User utterance: {utterance}\n")

    # Step 1 — Detect preferences
    print("🧩 Step 1 — Detecting preferences...")
    step1_raw = query_ollama("preference_step1", utterance)
    step1 = safe_json_parse(step1_raw, [])
    print(json.dumps(step1, indent=2))

    if not step1:
        print("\nNo preferences found, stopping pipeline.")
        return

    # Step 2 — Extract entities
    print("\n🧩 Step 2 — Extracting entities...")
    step2_input = json.dumps(step1, indent=2)
    step2_raw = query_ollama("preference_step2", step2_input)
    step2 = safe_json_parse(step2_raw, step1)
    print(json.dumps(step2, indent=2))

    # Step 3 — Convert to logic representation
    print("\n🧠 Step 3 — Converting to logic form...")
    step3_input = json.dumps(step2, indent=2)
    logic = query_ollama("preference_step3", step3_input)
    print(logic.replace("json", "").replace("`", ""))


if __name__ == "__main__":
    utterance = input("Enter a test utterance: ")
    run_pipeline(utterance)
