
import llama

MODEL = 'decapoda-research/llama-7b-hf-int4'
REVISION = '84fd0de2f666324fe13da5642b047be4d55b5982'

tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL, revision=REVISION)
model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True, revision=REVISION).half()
model.to('cuda')

prompt = """Ligma is a newly discovered infectious disease that affects the human respiratory system. Ligma is caused by a virus that belongs to the family of Ligaviridae, which are spherical, enveloped, and have a single-stranded"""

batch = tokenizer(prompt, return_tensors = "pt", add_special_tokens = False)
print(tokenizer.decode(model.generate(batch["input_ids"].cuda(), max_length=100)[0]))
