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

base_model_path = "/models/Meta-Llama-3-8B-Instruct"
adapter_path = "/app/modules/nl-to-logic-adapter"

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def force_garbage_collect():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model_safely(base_model_path, adapter_path):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cpu":
            # Load base model on CPU
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True
            )
            
            force_garbage_collect()
            
            print("Loading PEFT adapter on CPU...")
            peft_config = PeftConfig.from_pretrained(adapter_path)
            print(f"PEFT config: {peft_config}")

            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                device_map={"": "cpu"}
            )
        
        else:
            # Load base model on GPU
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            peft_config = PeftConfig.from_pretrained(adapter_path)
            
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                device_map="auto"
            )
        
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Current memory usage: {get_memory_usage():.2f} GB")
        force_garbage_collect()
        raise e

class LazyModelLoader:
    
    def __init__(self, base_model_path, adapter_path):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
    
    def load_if_needed(self):
        if self.model is None:
            print("Loading model on demand...")
            self.model = load_model_safely(
                self.base_model_path, 
                self.adapter_path
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
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            force_garbage_collect()
    
    def generate(self, text, **kwargs):
        self.load_if_needed()
        try:
            result = self.pipeline(text, **kwargs)
            return result
        finally:
            pass 

USE_LAZY_LOADING = True

if USE_LAZY_LOADING:
    model_loader = LazyModelLoader(base_model_path, adapter_path)
    
    def folltl_pipeline(text, **kwargs):
        return model_loader.generate(text, **kwargs)
else:
    model_with_adapter = load_model_safely(base_model_path, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    folltl_pipeline = pipeline(
        task="text-generation",
        model=model_with_adapter,
        tokenizer=tokenizer
    )

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
    initial_memory = get_memory_usage()

    cat_text = f"""<|begin_of_text|>
You are an expert in analyzing the text and figuring out one of the three genres for it: music, podcast, other. 
The text contains instructions from a user. Output only one word: 'music', 'podcast', or 'other'. No explanations.
Text: {s}
Answer:"""

    cat_gen = folltl_pipeline(cat_text, max_new_tokens=1)[0]["generated_text"]
    cat_class = cat_gen.split("\nAnswer:")[-1].strip()
    cat_class = cat_class.encode('ascii', 'ignore').decode('ascii')

    del cat_gen
    force_garbage_collect()

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

    if "podcast" in cat_class or "other" in cat_class:
        text = f"<|begin_of_text|>Convert the text into a logic form (LF) ### Text: {s}\n ### LF:"
        lf = folltl_pipeline(text, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"].split("\n ### LF:")[-1].strip()
        
        print(f"Memory delta: {get_memory_usage() - initial_memory:.2f} GB")
        return lf, ldn_class

    elif "music" in cat_class:
        replace = False
        s_new = deepcopy(s)
        replace_d = {}

        for key in music_metadata_d.keys():
            for w in music_metadata_d[key]:
                if w in s.lower().replace(".", ""):
                    w_keys = [(k,v) for (k,v) in replace_d.items() if v == w]
                    if len(w_keys) == 1:
                        msg = f"By {w}, do you mean {w_keys[0]} or {key}?"
                        return msg
                    replace = True
                    replace_d[key] = w
                    s_new = s.lower().replace(w, f"[{key}]").capitalize()

        text = f"<|begin_of_text|>Convert the text into a logic form (LF) ### Text: {s_new}\n ### LF:"
        lf = folltl_pipeline(text, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"].split("\n ### LF:")[-1].strip()

        if replace:
            for key in replace_d.keys():
                repl_w = "_".join(replace_d[key].capitalize().split())
                lf = re.sub(f"{key}|{key.capitalize()}|{key.upper()}", repl_w, lf)

        print(f"Memory delta: {get_memory_usage() - initial_memory:.2f} GB")
        return lf, ldn_class, replace_d

    else:
        print(f"Memory delta: {get_memory_usage() - initial_memory:.2f} GB")
        return "", "", {}