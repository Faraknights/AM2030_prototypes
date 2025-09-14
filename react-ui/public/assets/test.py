import json

# Define the prompt template
PROMPT_TEMPLATE = """
You are an assistant designed to extract structured data from user input.

Your task is to:
- Identify the correct `intent`
- Extract each expected parameter from the user sentence
- If a parameter is not clearly mentioned or implied, you must set its value to **null**

---

Output format (JSON):
"""

# Load intents.json
with open("intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)

# Iterate through each intent and print one prompt (no example)
for intent_entry in intents_data:
    intent_name = intent_entry.get("intent")
    parameters = intent_entry.get("parameters", [])

    print("=" * 40)
    print(PROMPT_TEMPLATE.strip())
    print(f"Intent: \"{intent_name}\"")
    print("\nYour output:\n")

    print("{")
    print(f'  "intent": "{intent_name}",')
    if parameters:
        print("  \"parameters\": {")
        for i, param in enumerate(parameters):
            name = param.get("name")
            typ = param.get("type", "unknown")
            desc = param.get("description", "").strip()
            comma = "," if i < len(parameters) - 1 else ""
            print(f'    "{name}": "{typ} | null"    #{desc}, null if unspecified{comma}')
        print("  }")
    else:
        print("  \"parameters\": {}")
    print("}\n")
