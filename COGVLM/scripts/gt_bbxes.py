import json

file_path = 'VLM-Grounding/COGVLM/data/annotations/instances_val2017.json'
output_file_path = 'VLM-Grounding/COGVLM/data/gt_bboxes/gt_bboxes.json'


with open(file_path, 'r') as file:
    data = json.load(file)


bounding_boxes_by_image = {}


for annotation in data['annotations']:
    image_id = annotation['image_id']
    bbox = annotation['bbox']
    if image_id not in bounding_boxes_by_image:
        bounding_boxes_by_image[image_id] = []
    bounding_boxes_by_image[image_id].append(bbox)


with open(output_file_path, 'w') as outfile:
    for image_id, bboxes in bounding_boxes_by_image.items():
        entry = json.dumps({"image_id": str(image_id), "bboxes": bboxes})
        outfile.write(entry + '\n')