cat_2017 = 'VLM-Grounding/COGVLM/data/annotations/instances_val2017.json'

import json

def main():
    json_file = cat_2017
    if json_file is not None:
        with open(json_file, 'r') as COCO:
            js = json.loads(COCO.read())
            # Extract and print only the 'name' field from each category
            category_names = [category['name'] for category in js['categories']]
            print(category_names)

if __name__ == "__main__":
    main()