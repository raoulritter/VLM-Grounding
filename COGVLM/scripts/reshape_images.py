import json
from PIL import Image

# Load the JSON data
with open('../data/bbxes_objects/bboxes_objects.json', 'r') as file:
    predicted_bboxes = json.load(file)

# Function to adjust bounding boxes according to actual image dimensions
def adjust_bboxes(bboxes, orig_size, target_size=(1000, 1000)):
    orig_width, orig_height = orig_size
    adjusted_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        # Scale the bounding box coordinates
        x = x / target_size[0] * orig_width
        y = y / target_size[1] * orig_height
        w = w / target_size[0] * orig_width
        h = h / target_size[1] * orig_height
        adjusted_bboxes.append([x, y, w, h])
    return adjusted_bboxes

# Initialize the new predicted bboxes data
new_predicted_bboxes = []

# Process each image
for item in predicted_bboxes:
    image_id = item['image'].split('.')[0]
    
    # Load the image to get its actual size
    image_path = f'../data/images/{image_id}.jpg'
    image = Image.open(image_path)
    orig_size = image.size
    
    # Extract and adjust bounding boxes from the data
    new_bbx_with_object = []
    for obj in item['bbx with object']:
        obj_name, bbox = obj.split(' [')
        bbox = bbox.strip(']').split(', ')
        bbox = [float(coord) for coord in bbox]
        adjusted_bbox = adjust_bboxes([bbox], orig_size)[0]
        new_bbx_with_object.append(f"{obj_name} [{', '.join(map(str, adjusted_bbox))}]")
    
    # Append the new data to the list
    new_predicted_bboxes.append({
        "image": item['image'],
        "bbx with object": new_bbx_with_object
    })

# Save the new predicted bboxes to a new JSON file
with open('../data/bbxes_objects/adjusted_bbox_objects.json', 'w') as file:
    json.dump(new_predicted_bboxes, file, indent=4)

print("Adjusted bounding boxes have been saved to 'adjusted_bbox_objects.json'.")
