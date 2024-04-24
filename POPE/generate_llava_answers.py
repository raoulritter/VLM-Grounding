import json
import argparse
import sys
sys.path.append('../LLaVA/')
from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model
from llava.mm_utils import get_model_name_from_path

def run_inference(args, image_path, question):
    model_name = get_model_name_from_path(args.model_path)
    model_args = type('Args', (), {
        "model_path": args.model_path,
        "model_base": None,
        "model_name": model_name,
        "query": question,
        "conv_mode": None,
        "image_file": image_path,
        "sep": args.sep,
        "temperature": args.temperature,
        "top_p": None,
        "num_beams": 1,
        "do_sample": True,
        "max_new_tokens": 512
    })()
    return eval_model(model_args)

def process_questions(args):
    results = []

    with open(args.input_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            image_path = args.image_dir + data['image']
            print("image path:", image_path)
            answer = run_inference(args, image_path, data['text'])
            print("answer: ", answer)
            results.append({'question': data['text'], 'answer': answer})
    
    with open(args.output_file, 'w') as out_file:
        for result in results:
            print("result: ", result)
            json.dump(result, out_file)
            out_file.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Run inference on images and questions')
    parser.add_argument('--input_file', type=str, default='output/coco/coco_pope_popular_10.json', help='Path to the input JSON file with questions and image names')
    parser.add_argument('--output_file', type=str, default='output/coco/ans_files/coco_pope_popular_10_llava_ans.json', help='Path to the output JSON file for storing answers')
    parser.add_argument('--image_dir', type=str, default='data/val2014/', help='Directory where images are stored')
    parser.add_argument('--model_path', type=str, default='liuhaotian/llava-v1.5-7b', help='Path to the model')
    parser.add_argument('--sep', type=str, default=',', help='Separator character used in the model arguments')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for model sampling')

    args = parser.parse_args()
    process_questions(args)

if __name__ == '__main__':
    main()
