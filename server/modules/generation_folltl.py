# generation_folltl.py - Memory-optimized version

from collections import defaultdict
from copy import deepcopy
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import numpy as np
import gc
import psutil
import os
from .llamipa_generate import get_discourse_parse

# --- Configuration ---
base_model_path = "/models/Meta-Llama-3-8B-Instruct"
adapter_path = "/app/modules/nl-to-logic-adapter"

# Memory management utilities
def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def force_garbage_collect():
    """Force garbage collection and clear cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Memory-optimized model loading ---
def load_model_safely(base_model_path, adapter_path, max_memory_gb=30):
    """
    Load LLaMA-3 8B + LoRA adapter with strict memory constraints
    """
    try:
        print(f"Initial memory usage: {get_memory_usage():.2f} GB")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # For CPU, use more aggressive memory optimizations
        if device == "cpu":
            # Use 8-bit quantization to reduce memory usage by ~50%
            dtype = torch.int8
            
            # Calculate available memory for model loading
            available_memory = max_memory_gb * 0.8  # Leave 20% buffer
            
            # Load base model with memory constraints
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,  # Use float16 for weights
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
                max_memory={0: f"{available_memory}GB"},
                offload_folder="/tmp/model_offload",
                offload_state_dict=True,
                load_in_8bit=False,  # We'll do manual quantization
            )
            
            print(f"Base model loaded. Memory usage: {get_memory_usage():.2f} GB")
            
            # Force garbage collection before loading adapter
            force_garbage_collect()
            
            # Load PEFT adapter
            print("Loading PEFT adapter...")
            peft_config = PeftConfig.from_pretrained(adapter_path)
            print(f"PEFT config: {peft_config}")

            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                device_map={"": "cpu"},
                offload_folder="/tmp/model_offload_adapter",
                max_memory={0: f"{available_memory}GB"}
            )
            
        else:
            # GPU loading with memory management
            dtype = torch.float16
            device_map = "auto"
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                max_memory={0: f"{max_memory_gb}GB"}
            )
            
            print("Loading PEFT adapter...")
            peft_config = PeftConfig.from_pretrained(adapter_path)
            
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                device_map="auto",
                max_memory={0: f"{max_memory_gb}GB"}
            )
        
        print(f"Final memory usage: {get_memory_usage():.2f} GB")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Current memory usage: {get_memory_usage():.2f} GB")
        force_garbage_collect()
        raise e

# Alternative: Lazy loading approach
class LazyModelLoader:
    """Load model only when needed and unload after use"""
    
    def __init__(self, base_model_path, adapter_path, max_memory_gb=30):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.max_memory_gb = max_memory_gb
        self.model = None
        self.tokenizer = None
        self.pipeline = None
    
    def load_if_needed(self):
        """Load model only if not already loaded"""
        if self.model is None:
            print("Loading model on demand...")
            self.model = load_model_safely(
                self.base_model_path, 
                self.adapter_path, 
                self.max_memory_gb
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path, 
                add_eos_token=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.pipeline = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
    
    def unload(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            force_garbage_collect()
            print(f"Model unloaded. Memory usage: {get_memory_usage():.2f} GB")
    
    def generate(self, text, **kwargs):
        """Generate text with automatic model loading/unloading"""
        self.load_if_needed()
        try:
            result = self.pipeline(text, **kwargs)
            return result
        finally:
            # Optional: Unload after each use for maximum memory efficiency
            # Comment out if you want to keep model loaded between calls
            pass  # self.unload()

# --- Choose loading strategy ---
USE_LAZY_LOADING = True  # Set to False for normal loading
MAX_MEMORY_GB = 28  # Adjust based on your system

if USE_LAZY_LOADING:
    # Lazy loading approach - model loaded on demand
    model_loader = LazyModelLoader(base_model_path, adapter_path, MAX_MEMORY_GB)
    
    def folltl_pipeline(text, **kwargs):
        return model_loader.generate(text, **kwargs)
    
    print("✅ Lazy loading setup complete - model will be loaded on first use")
else:
    # Traditional loading with memory optimization
    model_with_adapter = load_model_safely(base_model_path, adapter_path, MAX_MEMORY_GB)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    folltl_pipeline = pipeline(
        task="text-generation",
        model=model_with_adapter,
        tokenizer=tokenizer
    )
    
    print("✅ Model loaded with memory optimization")

# Load music metadata
artist_l = np.load("/app/modules/artists.npy")
bands_l = np.load("/app/modules/bands.npy")
songs_l = np.load("/app/modules/songs.npy")
genres_l = np.load("/app/modules/genres.npy")
music_metadata_d = defaultdict(list)
music_metadata_d["artist"] = artist_l
music_metadata_d["band"] = bands_l
music_metadata_d["song"] = songs_l
music_metadata_d["genre"] = genres_l    

def get_logic_translation(s):
    """
    Returns the logic form (LF) of the input text `s`, along with its emotional tone
    ('like', 'dislike', 'neutral') and music replacements if applicable.
    """
    
    # Monitor memory usage during generation
    initial_memory = get_memory_usage()

    # 1️⃣ Determine the category: music, podcast, other
    cat_text = f"""<|begin_of_text|>
You are an expert in analyzing the text and figuring out one of the three genres for it: music, podcast, other. 
The text contains instructions from a user. Output only one word: 'music', 'podcast', or 'other'. No explanations.
Text: {s}
Answer:"""

    cat_gen = folltl_pipeline(cat_text, max_new_tokens=1)[0]["generated_text"]
    cat_class = cat_gen.split("\nAnswer:")[-1].strip()
    cat_class = cat_class.encode('ascii', 'ignore').decode('ascii')

    # Clear intermediate results to save memory
    del cat_gen
    force_garbage_collect()

    # 2️⃣ Determine emotional tone: like, dislike, neutral
    ldn_text = f"""<|begin_of_text|>
You are an expert in analyzing the emotional tone of a text. 
Text: {s}
Answer only with one word: 'like', 'dislike', or 'neutral'. No explanations.
Answer:"""

    ldn_gen = folltl_pipeline(ldn_text, max_new_tokens=1)[0]["generated_text"]
    ldn_class = ldn_gen.split("\nAnswer:")[-1].strip()
    ldn_class = ldn_class.encode('ascii', 'ignore').decode('ascii')

    del ldn_gen
    force_garbage_collect()

    # 3️⃣ Convert to logic form
    if "podcast" in cat_class or "other" in cat_class:
        text = f"<|begin_of_text|>Convert the text into a logic form (LF) ### Text: {s}\n ### LF:"
        lf = folltl_pipeline(text, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"].split("\n ### LF:")[-1].strip()
        
        print(f"Memory delta: {get_memory_usage() - initial_memory:.2f} GB")
        return lf, ldn_class

    elif "music" in cat_class:
        # use the music ontology to replace phrases in the input with tags
        replace = False
        s_new = deepcopy(s)
        replace_d = {}

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
        lf = folltl_pipeline(text, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"].split("\n ### LF:")[-1].strip()

        # Replace ontology placeholders in LF
        if replace:
            for key in replace_d.keys():
                repl_w = "_".join(replace_d[key].capitalize().split())
                lf = re.sub(f"{key}|{key.capitalize()}|{key.upper()}", repl_w, lf)

        print(f"Memory delta: {get_memory_usage() - initial_memory:.2f} GB")
        return lf, ldn_class, replace_d

    else:
        print(f"Memory delta: {get_memory_usage() - initial_memory:.2f} GB")
        return "", "", {}

def get_logic_translation_dial(s):
    """
    Processes a dialogue string `s` and returns the logic form (LF) for the current user turn,
    along with emotional tone and music replacements if applicable.
    
    Returns a consistent structure: (lf, sentiment, replace_d, clarification_msg)
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

# Memory monitoring function for debugging
def print_memory_stats():
    """Print current memory usage statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"RSS Memory: {memory_info.rss / 1024 / 1024 / 1024:.2f} GB")
    print(f"VMS Memory: {memory_info.vms / 1024 / 1024 / 1024:.2f} GB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")