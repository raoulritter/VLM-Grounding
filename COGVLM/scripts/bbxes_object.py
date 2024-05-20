import json
import re

def extract_bboxes(caption):
    """ Extract bounding boxes from caption and convert to COCO format, handling multiple bounding boxes separated by semicolons. """
    bbox_pattern = r'\[\[([\d,;]+)\]\]'
    matches = re.findall(bbox_pattern, caption)
    bboxes = []
    for match in matches:
        # Split the match by semicolon in case of multiple bounding boxes
        individual_bboxes = match.split(';')
        for bbox in individual_bboxes:
            coords = list(map(int, bbox.split(',')))
            # Convert to COCO format: [x, y, width, height]
            bboxes.append([coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1], f"{coords[0]},{coords[1]},{coords[2]},{coords[3]}"])
    return bboxes

def load_json(filename):
    """ Load JSON data from a file where each line is a separate JSON object. """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def get_word_preceding_bbox(caption, bbox_text):
    """ Get the word preceding the bounding box text in the caption. """
    pattern = rf'(\w+)\s+\[\[{bbox_text}\]\]'
    match = re.search(pattern, caption)
    return match.group(1) if match else "unknown"

def process_entries(captions_file, objects_file):
    captions_data = load_json(captions_file)
    objects_data = load_json(objects_file)

    # Create a dictionary for quick access to objects by image_id
    objects_dict = {obj['image']: obj['objects'] for obj in objects_data}

    # Process each entry in captions data
    processed_data = []
    for entry in captions_data:
        image_id = entry['image_id']
        caption = entry['caption']
        bboxes = extract_bboxes(caption)
        objects = objects_dict.get(image_id, [])

        # Combine bboxes with objects
        combined_data = {
            "image": image_id,
            "bbx with object": []
        }

        # Handle mismatch in number of objects and bounding boxes
        min_length = min(len(bboxes), len(objects))
        for i in range(min_length):
            combined_data["bbx with object"].append(f"{objects[i]} {bboxes[i][:4]}")

        # Handle cases where there are more bounding boxes than objects
        for i in range(min_length, len(bboxes)):
            bbox_text = bboxes[i][4]
            preceding_word = get_word_preceding_bbox(caption, bbox_text)
            combined_data["bbx with object"].append(f"{preceding_word} {bboxes[i][:4]}")

        processed_data.append(combined_data)

    return processed_data

def save_processed_data(processed_data, output_file):
    with open(output_file, 'w') as file:
        json.dump(processed_data, file, indent=4)


captions_file = 'VLM-Grounding/COGVLM/data/captions/captions.json'
objects_file = 'VLM-Grounding/COGVLM/data/objects/objects.json'
output_file = 'VLM-Grounding/COGVLM/data/bbxes_objects/bbxes_objects.json'

processed_data = process_entries(captions_file, objects_file)
save_processed_data(processed_data, output_file)
