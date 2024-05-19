import torch
from sentence_transformers import SentenceTransformer
import json

cat_2017 = 'VLM-Grounding/COGVLM/data/annotations/instances_val2017.json'
## get coco classes
json_file = cat_2017
if json_file is not None:
    with open(json_file, 'r') as COCO:
        js = json.loads(COCO.read())
        # Extract and print only the 'name' field from each category
        class_list_coco = [category['name'] for category in js['categories']]


## get objects aka the classes that cogvlm predicts
unique_objects = set()
with open('VLM-Grounding/COGVLM/data/objects/objects.json', 'r') as file:
    for line in file:
        data = json.loads(line)
        # Add objects from each line to the set (automatically handles duplicates)
        unique_objects.update(data['objects'])        

final_class_list = list(unique_objects)        

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text):
    output = model.encode(text)
    return torch.tensor(output)

gt_classes = class_list_coco
predictions = final_class_list

gt_embeddings = {cls: get_embedding(cls) for cls in gt_classes}
prediction_embeddings = {pred: get_embedding(pred) for pred in predictions}

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0)

threshold1 = 0.7  # Adjust based on desired strictness
mapped_predictions1 = {}

for pred, pred_embedding in prediction_embeddings.items():
    similarities = {gt: cosine_similarity(pred_embedding, gt_embeddings[gt]) for gt in gt_classes}
    best_match, best_sim = max(similarities.items(), key=lambda x: x[1])
    if best_sim > threshold1:
        if best_match in mapped_predictions1:
            mapped_predictions1[best_match].append(pred)
        else:
            mapped_predictions1[best_match] = [pred]

with open('VLM-Grounding/COGVLM/data/classes/object_to_class.json', 'w') as file:
    json.dump(mapped_predictions1, file, indent=4)

