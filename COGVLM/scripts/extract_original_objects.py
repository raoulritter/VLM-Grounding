import json
import re
import inflect

# Initialize inflect engine for singularization
p = inflect.engine()

# List of allowed COCO classes
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Function to process each caption and extract bounding boxes and associated words
def process_caption(caption, filter_non_coco=False):
    pattern = r'\[\[(.*?)\]\]'
    matches = re.findall(pattern, caption)
    processed_caption = re.sub(pattern, '[BBX]', caption)
    words = processed_caption.split()
    result = []

    bbx_index = 0
    for word in words:
        if '[BBX]' in word:
            bbx_data = matches[bbx_index].split(';')
            singular_word = p.singular_noun(previous_word) or previous_word
            if not filter_non_coco or singular_word in coco_classes:
                for bbx in bbx_data:
                    coords = list(map(int, bbx.split(',')))
                    bbx_str = f"{singular_word} [{coords[0]}, {coords[1]}, {coords[2] - coords[0]}, {coords[3] - coords[1]}]"
                    result.append(bbx_str)
            bbx_index += 1
        previous_word = word.replace('[BBX]', '').strip()
    
    return result

# Read the input file
input_file = 'captions.json'
with open(input_file, 'r') as file:
    data = file.readlines()

# Process each entry
output_all = []
output_filtered = []

for entry in data:
    entry = json.loads(entry)
    image_id = entry['image_id']
    caption = entry['caption']
    
    # Process caption without filtering
    processed_bbx_all = process_caption(caption, filter_non_coco=False)
    if processed_bbx_all:
        output_entry_all = {
            "image": image_id,
            "bbx with object": processed_bbx_all
        }
        output_all.append(output_entry_all)
    
    # Process caption with filtering
    processed_bbx_filtered = process_caption(caption, filter_non_coco=True)
    if processed_bbx_filtered:
        output_entry_filtered = {
            "image": image_id,
            "bbx with object": processed_bbx_filtered
        }
        output_filtered.append(output_entry_filtered)

# Write the output to a new JSON file
output_file_all = '../data/bboxes_objects/bboxes_original_objects.json'
output_file_filtered = '../data/bboxes_objects/bboxes_original_removed.json'

with open(output_file_all, 'w') as file:
    json.dump(output_all, file, indent=4)

with open(output_file_filtered, 'w') as file:
    json.dump(output_filtered, file, indent=4)

print("Processing complete. Outputs saved to", output_file_all, "and", output_file_filtered)

