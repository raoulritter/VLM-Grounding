import json

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def count_objects_per_image(data):
    results = []
    total_objects = 0

    for item in data:
        image = item['image']
        object_count = len(item['bbx with object'])
        total_objects += object_count

        results.append({
            "image": image,
            "object_count": object_count
        })

    final_result = {
        "images": results,
        "total_object_count": total_objects
    }

    return final_result

def save_to_json(result, filename):
    with open(filename, 'w') as json_file:
        json.dump(result, json_file, indent=4)

def main(input_filename, output_filename):
    data = load_json(input_filename)
    result = count_objects_per_image(data)
    save_to_json(result, output_filename)

if __name__ == "__main__":
    input_filename = '../data/bboxes_objects/map_classes/bboxes_objects_0.7.json'
    output_filename = '../data/object_counts/map_classes_0.7.json'

    main(input_filename, output_filename)
