import json
import re

def convert_bbox_cogvlm_to_coco(cogvlm_bbox):
    x1, y1, x2, y2 = cogvlm_bbox
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    return [x, y, w, h]


with open('VLM-Grounding/COGVLM/data/captions/captions.json', 'r') as file:
    captions = [json.loads(line) for line in file]

bbox_pattern = r'\[\[(\d+,\d+,\d+,\d+)\]\]'

for caption_entry in captions:
    caption = caption_entry['caption']
    image_filename = caption_entry['image_id']
    bboxes = re.findall(bbox_pattern, caption)
    bboxes = [list(map(int, bbox.split(','))) for bbox in bboxes]
    coco_bbox = convert_bbox_cogvlm_to_coco(bboxes)

