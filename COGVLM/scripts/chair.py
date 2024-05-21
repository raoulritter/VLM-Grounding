import json

# Load the objects data
with open('VLM-Grounding/COGVLM/data/objects/objects.json', 'r') as f:
    objects_data = [json.loads(line) for line in f]

# Create a dictionary to map image IDs to objects
image_objects = {item['image']: item['objects'] for item in objects_data}

# Function to extract objects from a caption using the objects.json data
def extract_objects(image_id):
    return image_objects.get(image_id, [])

# Load the captions data
with open('VLM-Grounding/COGVLM/data/captions/captions.json', 'r') as f:
    captions_data = [json.loads(line) for line in f]

with open('VLM-Grounding/COGVLM/data/output/hallucinations.json', 'r') as file:
    hallucinations_data = json.load(file)    

# Variables to store counts
total_objects = 0
hallucinated_objects = 0
total_captions = len(captions_data)
captions_with_hallucinations = 0
total_caption_length = 0

# Process each caption
for caption_entry in captions_data:
    image_id = caption_entry['image_id']
    caption = caption_entry['caption']
    objects_in_caption = extract_objects(image_id)
    total_objects += len(objects_in_caption)
    hallucination_found = False

    # Calculate the length of the caption
    caption_length = len(caption.split())
    total_caption_length += caption_length

    for hallucination_entry in hallucinations_data:
        if image_id == hallucination_entry['image_id']:
            # Count each pred_box as one hallucinated object
            if 'pred_box' in hallucination_entry:
                hallucinated_objects += 1
                hallucination_found = True

    if hallucination_found:
        captions_with_hallucinations += 1

# Calculate CHAIRI and CHAIRS
CHAIRI = hallucinated_objects / total_objects if total_objects else 0
CHAIRS = captions_with_hallucinations / total_captions if total_captions else 0

# Calculate average caption length (Len)
average_caption_length = total_caption_length / total_captions if total_captions else 0

print("CHAIRI:", CHAIRI * 100)
print("CHAIRS:", CHAIRS * 100)
print("Len:", average_caption_length)