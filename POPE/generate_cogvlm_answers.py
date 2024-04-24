import json
import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

def load_model_and_tokenizer(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.local_tokenizer)
    torch_type = torch.bfloat16 if args.bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.from_pretrained,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=bool(args.quant),
        trust_remote_code=True
    ).to(args.device).eval()

    return model, tokenizer

def process_questions(args, model, tokenizer):
    results = []
    with open(args.input_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            image_path = args.image_dir + entry["image"]
            question = entry["text"]
            
            try:
                image = Image.open(image_path).convert('RGB')
            except FileNotFoundError:
                print(f"Error: Image file not found at path {image_path}")
                continue  

            input_by_model = model.build_conversation_input_ids(tokenizer, query=question, history=[], images=[image])
            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(args.device),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(args.device),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(args.device),
                'images': [input_by_model['images'][0].to(args.device).to(torch.bfloat16)],  
            }
            
            if 'cross_images' in input_by_model and input_by_model['cross_images']:
                inputs['cross_images'] = [input_by_model['cross_images'][0].to(args.device).to(torch.bfloat16)]

            gen_kwargs = {"max_length": 2048, "do_sample": False} 

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
            
                response = tokenizer.decode(outputs[0]).split("</s>")[0]
                
            results.append({'question_id': entry["question_id"], 'question': question, 'answer': response})
            print("response: ", response)

    with open(args.output_file, 'w') as out_file:
        for result in results:
            json.dump(result, out_file)
            out_file.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Process questions using a conversational model with image context")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file with questions and image names")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file for storing answers")
    parser.add_argument("--image_dir", type=str, default='/home/jwiers/POPE/data/val2014/', help="Directory where images are stored")
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help="Model identifier for pretrained model")
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help="Tokenizer identifier")
    parser.add_argument("--quant", type=int, choices=[4], default=None, help="Enable quantization (4-bit)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the model on")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    process_questions(args, model, tokenizer)

if __name__ == "__main__":
    main()
