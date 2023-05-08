import torch
from transformers import AutoTokenizer
from einops._torch_specific import allow_ops_in_compiled_graph

allow_ops_in_compiled_graph()

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('conceptofmind/PaLM', 'palm_410m_8k_v0').to(device)

opt_model = torch.compile(model, backend='hidet') 

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

encoded_text = tokenizer("My dog is very cute", return_tensors="pt")

output_tensor = opt_model.generate(
    256, 
    encoded_text['input_ids'].to(device),
)

decoded_output = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)
print(decoded_output)