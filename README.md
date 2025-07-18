# forread
for read



torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=1 \
  --master_addr="192.168.1.10" \
  --master_port=29500 \
  dp.py





  import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# DeepSpeed initialize
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

prompt = "O'zbekistonning poytaxti nima?"
inputs = tokenizer(prompt, return_tensors="pt").to(model_engine.device)

with torch.no_grad():
    outputs = model_engine.generate(
        **inputs,
        max_new_tokens=50
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))




{
    "train_batch_size": 2,
    "fp16": {
      "enabled": true
    },
    "zero_optimization": {
      "stage": 2
    }
  }

  ####################################################################################################################################################################
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128


pip install deepspeed
python -m deepspeed.env_report
pip install transformers
huggingface-cli login
nima_hf_MrHQDeldqsxQlQSfosngJqaHYvTUzrYQEr

##########################################################################################################################################################################
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import deepspeed

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

# DeepSpeed initialize
model = deepspeed.init_inference(
    model,
    mp_size=1,  # multi-GPU uchun oshirish mumkin
    dtype=torch.float16,
    replace_method="auto"
)


is_exit = True
while is_exit:
    prompt = input("User: ")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    if prompt.lower() == "exit":
        is_exit = False
##############################################################################################################################################################################
ray stop
ray start --address='172.16.8.38:6379'


#########################################################################################################################################
python -c "import torch; print(torch.__version__); print(torch.cuda.nccl.version())"

#########################################################################################################################################
export GLOO_SOCKET_IFNAME=enp130s0




