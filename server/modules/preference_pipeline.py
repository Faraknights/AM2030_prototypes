import json
import subprocess
from pathlib import Path

MODEL = "gemma3:4b"
PROMPT_DIR = Path("server/modules/prompts")

def load_prompt(filename: str) -> str:
    """Load a prompt template from file."""
    path = PROMPT_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def query_ollama(prompt: str, model: str = MODEL) -> str:
    """Run a local Ollama model with a given prompt."""
    cmd = ["ollama", "run", model]
    result = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True)
    return result.stdout.decode("utf-8").strip()


def safe_json_parse(text: str, fallback):
    """Safely parse JSON output."""
    try:
        return json.loads(text.replace("json", "").replace("`", ""))
    except json.JSONDecodeError:
        print("⚠️ JSON parsing failed. Raw output:")
        print(text)
        return fallback

# Pipeline steps
def step1_detect_preferences(utterance: str):
    prompt = load_prompt("preference_step1.txt").replace("{user_utterance}", utterance)
    output = query_ollama(prompt)
    return safe_json_parse(output, [])


def step2_extract_entities(step1_output: list):
    step1_json = json.dumps(step1_output, indent=2)
    prompt = load_prompt("preference_step2.txt").replace("{step1_output}", step1_json)
    output = query_ollama(prompt)
    return safe_json_parse(output, step1_output)


def step3_to_logic(step2_output: list):
    step2_json = json.dumps(step2_output, indent=2)
    prompt = load_prompt("preference_step3.txt").replace("{step2_output}", step2_json)
    return query_ollama(prompt).replace("json", "").replace("`", "")


# Main pipeline
def run_pipeline(utterance: str):
    print(f"\n🎯 User utterance: {utterance}\n")

    step1 = step1_detect_preferences(utterance)
    print("🧩 Step 1 — Preferences detected:")
    print(json.dumps(step1, indent=2))

    if not step1:
        print("\nNo preferences found, stopping pipeline.")
        return

    step2 = step2_extract_entities(step1)
    print("\n🧩 Step 2 — Entities extracted:")
    print(json.dumps(step2, indent=2))

    logic = step3_to_logic(step2)
    print("\n🧠 Step 3 — Logic form:")
    print(logic)


# Run
if __name__ == "__main__":
    utterance = input("Enter a test utterance: ")
    run_pipeline(utterance)
