import json

# Load the bounding boxes and object names
with open('../data/bboxes_objects/adjusted_bboxes_original_objects.json', 'r') as f:
    bboxes_objects = json.load(f)

# Load the class mapping
with open('../data/classes/object_to_class_0.7.json', 'r') as f:
    object_to_class = json.load(f)

# Reverse the mapping for easier lookup
class_mapping = {}
for coco_class, objects in object_to_class.items():
    for obj in objects:
        class_mapping[obj] = coco_class

# Map objects to their corresponding COCO classes
mapped_bboxes_objects = []
for entry in bboxes_objects:
    image = entry['image']
    bbx_with_object = entry['bbx with object']
    new_bbx_with_object = []

    for item in bbx_with_object:
        object_name, bbox = item.split(' [', 1)
        bbox = '[' + bbox
        if object_name in class_mapping:
            coco_class = class_mapping[object_name]
            new_bbx_with_object.append(f"{coco_class} {bbox}")

    if new_bbx_with_object:
        mapped_bboxes_objects.append({
            "image": image,
            "bbx with object": new_bbx_with_object
        })

# Write the mapped bounding boxes and objects to a new JSON file
with open('../data/bboxes_objects/map_classes/bboxes_objects_0.7.json', 'w') as f:
    json.dump(mapped_bboxes_objects, f, indent=4)

print("Mapping completed and JSON file saved.")
