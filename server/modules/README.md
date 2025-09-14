The repo contains the QLORA adapter of Llama-3-8B-Instruct finetuned to translate natural language to FOL/LTL.

**Instructions**

1. Download the pretrained LLM (https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
2. Add the path of pretrained LLM and the output file in generation_folltl.py under pretrained_model, and f respectively.
3. Modify "base_model_name_or_path" in adapter_config.json.
4. generation_folltl.py can now be used to get the logic translation for single user utterances as well as dialogue.
5. For single instruction, use the get_logic_translation function defined in generation_folltl.py. See lines 152-162 as an example.
6. For a dialogue, the input should be formatted in a certain way as shown in lines 166, 173, 179 of generation_folltl.py. Once formatted, use the get_logic_translation_dial function. This function internally calls the llamipa model and the behavior of the logic translator is governed by the output from llamipa. See lines 164-183 for three examples and their corresponding outputs.
