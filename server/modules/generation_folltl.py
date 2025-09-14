# generation_folltl.py - Fixed version

from collections import defaultdict
from copy import deepcopy
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import numpy as np
from .llamipa_generate import get_discourse_parse

# --- Configuration ---
base_model_path = "/models/Meta-Llama-3-8B-Instruct"
adapter_path = "/app/modules/nl-to-logic-adapter"

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# --- Fonction de chargement sûr ---
def load_model_safely(base_model_path, adapter_path):
    """Charge le modèle LLaMA-3 8B + LoRA adapter selon la disponibilité GPU"""
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=dtype,
            device_map="auto",
            offload_buffers=True,
            offload_folder="/tmp/model_offload",
            low_cpu_mem_usage=True
        )

        print("Loading PEFT adapter...")
        peft_config = PeftConfig.from_pretrained(adapter_path)
        print(f"PEFT config: {peft_config}")

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map="auto",
            offload_buffers=True,
            offload_folder="/tmp/model_offload"
        )
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# --- Chargement ---
model_with_adapter = load_model_safely(base_model_path, adapter_path)

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 1. General-purpose pipeline
folltl_pipeline = pipeline(
    task="text-generation",
    model=model_with_adapter,
    tokenizer=tokenizer
)

# --- Test ---
test_input = "I really like the new song of Lady Gaga"
output = folltl_pipeline(test_input,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7)
print(output)

print("✅ Both folltl_pipeline and logic_pipeline are ready")

artist_l = np.load("/app/modules/artists.npy")
bands_l = np.load("/app/modules/bands.npy")
songs_l = np.load("/app/modules/songs.npy")
genres_l = np.load("/app/modules/genres.npy")
music_metadata_d = defaultdict(list)
music_metadata_d["artist"] = artist_l
music_metadata_d["band"] = bands_l
music_metadata_d["song"] = songs_l
music_metadata_d["genre"] = genres_l    
    
from copy import deepcopy
import re

def get_logic_translation(s):
    """
    Returns the logic form (LF) of the input text `s`, along with its emotional tone
    ('like', 'dislike', 'neutral') and music replacements if applicable.
    """

    # 1️⃣ Determine the category: music, podcast, other
    cat_text = f"""<|begin_of_text|>
You are an expert in analyzing the text and figuring out one of the three genres for it: music, podcast, other. 
The text contains instructions from a user. Output only one word: 'music', 'podcast', or 'other'. No explanations.
Text: {s}
Answer:"""

    cat_gen = folltl_pipeline(cat_text, max_new_tokens=20, do_sample=True, temperature=0.7)[0]["generated_text"]
    cat_class = cat_gen.split("\nAnswer:")[-1].strip()
    cat_class = cat_class.encode('ascii', 'ignore').decode('ascii')  # remove unwanted characters

    # 2️⃣ Determine emotional tone: like, dislike, neutral
    ldn_text = f"""<|begin_of_text|>
You are an expert in analyzing the emotional tone of a text. 
Text: {s}
Answer only with one word: 'like', 'dislike', or 'neutral'. No explanations.
Answer:"""

    ldn_gen = folltl_pipeline(ldn_text, max_new_tokens=20, do_sample=True, temperature=0.7)[0]["generated_text"]
    ldn_class = ldn_gen.split("\nAnswer:")[-1].strip()
    ldn_class = ldn_class.encode('ascii', 'ignore').decode('ascii')  # remove unwanted characters

    # 3️⃣ Convert to logic form
    if "podcast" in cat_class or "other" in cat_class:
        text = f"<|begin_of_text|>Convert the text into a logic form (LF) ### Text: {s}\n ### LF:"
        lf = folltl_pipeline(text, max_new_tokens=100)[0]["generated_text"].split("\n ### LF:")[-1].strip()
        return lf, ldn_class

    elif "music" in cat_class:
        # use the music ontology to replace phrases in the input with tags like [band], [artist], [genre], [song]
        replace = False
        s_new = deepcopy(s)
        replace_d = {}
        conf_keys = {}

        for key in music_metadata_d.keys():
            for w in music_metadata_d[key]:
                if w in s.lower().replace(".", ""):
                    # Check for conflicting keys
                    w_keys = [(k,v) for (k,v) in replace_d.items() if v == w]
                    if len(w_keys) == 1:
                        msg = f"By {w}, do you mean {w_keys[0]} or {key}?"
                        return msg
                    replace = True
                    replace_d[key] = w
                    s_new = s.lower().replace(w, f"[{key}]").capitalize()

        text = f"<|begin_of_text|>Convert the text into a logic form (LF) ### Text: {s_new}\n ### LF:"
        lf = folltl_pipeline(text, max_new_tokens=100)[0]["generated_text"].split("\n ### LF:")[-1].strip()

        # Replace ontology placeholders in LF
        if replace:
            for key in replace_d.keys():
                repl_w = "_".join(replace_d[key].capitalize().split())
                lf = re.sub(f"{key}|{key.capitalize()}|{key.upper()}", repl_w, lf)

        return lf, ldn_class, replace_d

    else:
        # fallback
        return "", "", {}




def get_logic_translation_dial(s):
    """
    Processes a dialogue string `s` and returns the logic form (LF) for the current user turn,
    along with emotional tone and music replacements if applicable.
    
    Returns a consistent structure: (lf, sentiment, replace_d, clarification_msg)
    - lf: logic form string (empty if ignored)
    - sentiment: 'like', 'dislike', 'neutral' (empty if ignored)
    - replace_d: dictionary of music replacements (empty if none)
    - clarification_msg: clarification question string if needed, else None
    """

    # 1️⃣ Preprocess lines: add numbering
    lines = s.split("\n")
    for i in range(len(lines)):
        if not lines[i].startswith(str(i)):
            lines[i] = str(i+1) + " " + lines[i]

    # Keep last 10 lines for context window
    lines = lines[-10:]
    s = "\n".join(lines)

    # Prepend system start if missing
    if not s.startswith("0 <Assistant> Mission has started."):
        s = "0 <Assistant> Mission has started.\n" + s
    lines = s.split("\n")

    # 2️⃣ Identify dialogue turn boundaries
    inds = [
        i for i in range(1, len(lines))
        if ("<User>" in lines[i] and "<Assistant>" in lines[i-1]) or
           ("<User>" in lines[i-1] and "<Assistant>" in lines[i])
    ]

    # Build dialogue samples for discourse parsing
    samples_l = ["NEW DIALOGUE"]
    for i in range(len(inds)):
        if i < len(inds) - 1:
            samples_l.append(
                "Context: " + "\n".join(lines[:inds[i]]) +
                "\nStructure: \nNew Turn: " + "\n".join(lines[inds[i]:inds[i+1]])
            )
        else:
            samples_l.append(
                "Context: " + "\n".join(lines[:inds[i]]) +
                "\nStructure: \nNew Turn: " + "\n".join(lines[inds[i]:])
            )

    # 3️⃣ Discourse parsing
    parse = get_discourse_parse(samples_l)
    curr_idx = int(lines[-1].split()[0])

    # 4️⃣ Handle ignored turns and corrections
    ign_idxs = []
    corr_idx = None
    for part in parse.split():
        parent_idx = None
        nums = re.findall(r"\d+", part)
        if "COM" in part:
            if len(nums) > 1:
                ign_idxs.append(int(nums[1]))
        if "CONTIN" in part:
            if len(nums) > 0:
                parent_idx = int(nums[0])
        if parent_idx is not None and parent_idx in ign_idxs and len(nums) > 1:
            ign_idxs.append(int(nums[1]))
        if "CORR" in part and len(nums) > 1 and int(nums[1]) == curr_idx:
            corr_idx = int(nums[0])

    # 5️⃣ Decide what to process
    if curr_idx in ign_idxs:
        return "", "", {}, "Ignoring the current user utterance as it's a comment or continuation."
    
    elif corr_idx is not None:
        # Process corrected previous turn
        user_text = lines[corr_idx].split("<User> ")[1]
        out = get_logic_translation(user_text)
        if len(out) == 3:
            lf, sentiment, replace_d = out
            # Check if current turn has conflicting music terms
            curr_text = lines[curr_idx].split("<User> ")[1]
            for key in replace_d.keys():
                for w in music_metadata_d[key]:
                    if w in curr_text and "like" in sentiment:
                        clar_q = f"Do you like {replace_d[key]} or {w}?"
                        return lf, sentiment, replace_d, clar_q
            return lf, sentiment, replace_d, None
        else:
            return out[0], out[1], {}, None
    
    else:
        # Normal user turn
        user_text = lines[curr_idx].split("<User> ")[1]
        out = get_logic_translation(user_text)
        if len(out) == 3:
            return out[0], out[1], out[2], None
        else:
            return out[0], out[1], {}, None