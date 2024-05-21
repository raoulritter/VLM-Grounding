import json

file_path = '../data/annotations/instances_val2017.json'
output_file_path = '../data/gt_bboxes/gt_bboxes_new.json'

# Load your JSON data (assuming it's stored in a variable named 'data')
with open(file_path, 'r') as file:
    data = json.load(file)

# Create a dictionary to map category IDs to names
category_map = {category['id']: category['name'] for category in data['categories']}

# Process annotations to get the desired format
output = {}
for annotation in data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']
    
    # Format the bounding box with the object name
    bbox_with_object = f"{category_map[category_id]} [{', '.join(map(str, bbox))}]"
    
    # Use the image ID as the key and append bbox_with_object to the list
    if image_id not in output:
        output[image_id] = []
    output[image_id].append(bbox_with_object)

# Convert the processed data into the desired output format
result = []
for image_id, bbx_objects in output.items():
    result.append({
        "image": f"{str(image_id).zfill(12)}.jpg",
        "bbx with object": bbx_objects
    })

# Save the result to a new JSON file
with open(output_file_path, 'w') as file:
    json.dump(result, file, indent=4)

print("Transformation complete. The output has been saved to 'output.json'.")