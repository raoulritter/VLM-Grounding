import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import os
import json
from tqdm import tqdm

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-grounding-generalist-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()

query = 'Can you provide a detailed description of the image and include the coordinates [[x0,y0,x1,y1]] for each mentioned object?'


image_dir = '../data/images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

captions = []


for image_file in tqdm(image_files[:500], desc="Processing Images"):
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert('RGB')
    inputs = model.build_conversation_input_ids(tokenizer, query=query, images=[image])
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": False}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        caption = tokenizer.decode(outputs[0])
        #print(f"Caption for {image_file}: {caption}")

    captions.append({'image_id': image_file, 'caption': caption})


with open('../data/captions/captions.json', 'w') as f:
    for caption in captions:
        json.dump(caption, f)
        f.write('\n')

