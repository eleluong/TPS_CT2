# Load model directly
from transformers import AutoTokenizer
import torch 
import ctranslate2
import gradio as gr
import time

tokenizer_path = "meta-llama/Llama-2-7b-chat-hf"

# model_path = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path
)

model_path = "./llama2_7b_chat_ct2"
generator = ctranslate2.Generator(model_path, device = "cuda", compute_type = "int8_float16", inter_threads = 20, device_index = [0, 1, 2, 3])

tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

SYSTEM_PROMPT = """"""
TEMPLATE = """[INST] <<SYS>> <</SYS>> [INST] $$QUERY$$ [/INST]"""
sample_context = """"""
def infer_(query = ""):
    start_time = time.time()
    prompt = TEMPLATE.replace("$$QUERY$$", query)

    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
    outputs = generator.generate_batch([input_tokens],
                                        sampling_topp = 0.6,
                                        include_prompt_in_result = False, 
                                        max_length = len(input_tokens[0])+1000, 
                                        no_repeat_ngram_size = 10,
                                       )
    seqs = []
    target = outputs[0].sequences[0]
    response = tokenizer.decode(tokenizer.convert_tokens_to_ids(target), skip_special_tokens=True)

    response = response.replace("[/INST]", "")
    print(response)
    total_time = time.time() - start_time
    print("total: ", total_time)
    new_tokens = len(target)
    tps = new_tokens/total_time
    return response, tps

# demo = gr.Interface(fn = infer_, 
#                     inputs = [
#                         gr.Textbox(label = "Query"),
#                     ], 
#                     outputs = [
#                         gr.Textbox(label = "Answer"),
#                         gr.Textbox(label = "Speed")
#                     ])
# demo.queue()
# demo.launch(share = False)
