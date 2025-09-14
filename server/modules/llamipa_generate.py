import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

# Pretrained model name
PRETRAINED_MODEL = "/models/Meta-Llama-3-8B-Instruct"
# Fine-tuned model name
ADAPTER_PATH = "/app/modules/llamipa_adapter"

try:
    # Option 1: Try loading without device_map first
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    print("Loading PEFT adapter...")
    # Load PEFT config to check compatibility
    peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
    print(f"PEFT config: {peft_config}")
    
    # Load adapter with low_cpu_mem_usage for faster loading
    model = PeftModel.from_pretrained(
        model, 
        ADAPTER_PATH,
        low_cpu_mem_usage=True
    )
    
except Exception as e:
    print(f"Failed with first approach: {e}")
    print("Trying alternative approach...")
    
    # Option 2: Alternative loading approach
    try:
        model = AutoModelForCausalLM.from_pretrained(
            PRETRAINED_MODEL,
            device_map="cpu",  # Force CPU loading first
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        model = PeftModel.from_pretrained(
            model, 
            ADAPTER_PATH,
            device_map="auto"  # Let PEFT handle device mapping
        )
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")
        
        # Option 3: Last resort - load without any device mapping
        print("Trying CPU-only loading...")
        model = AutoModelForCausalLM.from_pretrained(
            PRETRAINED_MODEL,
            dtype=torch.float16
        )
        
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, add_eos_token=True)
tokenizer.pad_token_id = tokenizer.eos_token_id + 1
tokenizer.padding_side = "right"

# Pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)

print("✅ Model with adapter loaded successfully")
print("Device map:", model.hf_device_map if hasattr(model, 'hf_device_map') else "Not available")

# --- Utility functions remain the same ---

def check_endpoints(struct, head):
    """
    takes a struct string and a head int and returns only 
    the struct rels with sources that are >= head
    """
    new_rels_list = []
    new_rels = None
    if struct:
        rels = struct.split(' ')
        for rel in rels:
            if len(rel) > 0:
                source = int(rel.split('(')[1].split(',')[0].strip())
                if source >= head:
                    new_rels_list.append(rel)
        if len(new_rels_list) > 0:
            new_rels = ' '.join(new_rels_list)
    return new_rels

def add_previous(sample, previous, predictions):
    new_output = []
    keep_str = None
    #get head
    slist = sample.split('\n')
    head = int(slist[0].split('Context:')[1].split('<')[0].strip())
    # check current structure
    for s in slist:
        if s.startswith('Structure:'):
            new_structure = check_endpoints(previous, head)
            if new_structure:
                s = 'Structure: ' + new_structure + ' ' + predictions
                keep_str = new_structure + ' ' + predictions
            else:
                s = 'Structure: ' + predictions
                keep_str = predictions
        new_output.append(s)
    new_output_string = '\n'.join(new_output)
    return keep_str, new_output_string

def format_gen(preds):
    labels = ['COM','CONTR','CORR','QAP','ACK','ELAB','CLARIFQ','COND','CONTIN',
              'RES','EXPL','QELAB','ALT','NARR','CONFQ','SEQ']
    split_list = [st.strip() for st in preds.split(' ')]
    clean_list = []
    for a in split_list:
        s_tuple = None
        rel = None
        try:
            s = a.split('(')[1].split(')')[0].split(',')
            r = a.split('(')[0].strip()
        except IndexError:
            print('split error one')
        else:
            try:
                s_tuple = (int(s[0]), int(s[1]))
            except IndexError:
                print('split error two')
            except ValueError:
                print('value error three')
            if r in labels:
                #make sure the label is well-formed 
                rel = r
        if rel != None and s_tuple != None:
            clean_list.append(rel + '(' + str(s_tuple[0]) + ',' + str(s_tuple[1]) + ')')
    clean_preds = ' '.join(clean_list)
    return clean_preds


def formatting_prompts_func(example):
    output_text = '<|begin_of_text|>Identify the discourse structure (DS) for the new turn in the following excerpt :\n ' + example + '\n ### DS:'
    return output_text

def get_discourse_parse(samples_l):
    new_generations = ""
    previous_generations = ""
    new_dialogue = False

    for datum in samples_l:
        if datum.startswith('NEW DIALOGUE'):
            print("cond satisfied")
            new_dialogue = True
            continue

        if new_dialogue:
            # First example in new dialogue
            text = formatting_prompts_func(datum)
            previous_generations = ""
            new_dialogue = False
        else:
            # Make sure head, edu, and relations match up
            update_prev, amended_text = add_previous(datum, previous_generations, new_generations)
            previous_generations = update_prev or ""
            text = formatting_prompts_func(amended_text)

        # Generate discourse structure
        try:
            generated = pipe(text)[0].get('generated_text', "")
        except Exception as e:
            print(f"Error during generation: {e}")
            generated = ""

        # Safely extract new generations
        ds_split = generated.split('### DS:')
        new_gen = ds_split[1] if len(ds_split) > 1 else ""
        new_generations = format_gen(new_gen) or ""

    # Ensure safe concatenation even if one part is empty
    final_parse = (previous_generations or "") + " " + (new_generations or "")
    return final_parse.strip()
