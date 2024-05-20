import json

# Load the existing objects.json file
with open('VLM-Grounding/COGVLM/data/objects/objects.json', 'r') as f:
    objects_data = [json.loads(line) for line in f]

# Filter for entries with empty objects arrays
empty_objects_ids = [entry['image'] for entry in objects_data if not entry['objects']]

# Save the IDs to a new JSON file
with open('VLM-Grounding/COGVLM/data/objects/empty_objects_ids.json', 'w') as f:
    json.dump(empty_objects_ids, f, indent=4)

print(f"Found {len(empty_objects_ids)} images with empty objects.")