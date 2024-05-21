import json
import re

# def extract_bboxes(caption):
#     """ Extract bounding boxes from caption and convert to COCO format. """
#     bbox_pattern = r'\[\[(\d+),(\d+),(\d+),(\d+)\]\]'
#     matches = re.findall(bbox_pattern, caption)
#     bboxes = [[int(x), int(y), int(x2) - int(x), int(y2) - int(y)] for x, y, x2, y2 in matches]
#     return bboxes

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
            bboxes.append([coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]])
    return bboxes

def load_json(filename):
    """ Load JSON data from a file where each line is a separate JSON object. """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data
    

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
            combined_data["bbx with object"].append(f"{objects[i]} {bboxes[i]}")

        # Handle cases where there are more bounding boxes than objects
        # for i in range(min_length, len(bboxes)):
        #     combined_data["bbx with object"].append(f"unknown {bboxes[i]}")

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