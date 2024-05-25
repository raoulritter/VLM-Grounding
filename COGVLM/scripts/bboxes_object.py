import json
import re

# Create a dictionary mapping synonyms to categories
synonym_to_category = {
    'person': ['girls', 'men', 'beard', 'bun', 'bride', 'man', 'women', 'head', 'person', 'baby', 'crowd', 'boy', 'toddler', 'People', 'player', 'skateboarder', 'skier', 'passengers', 'surfer', 'groom', 'child', 'friends', 'Sailors', 'team', 'family', 'players', 'teammates', 'girl', 'bikinis', 'shoulder', 'children', 'faces', 'boys', 'expression', 'mouth', 'hair', 'coach', 'boys', 'people', 'groom', 'woman', 'eye', 'face', 'gentleman', 'player', 'walker', 'drinker', 'shopper', 'hunter', 'villager', 'worker', 'male', 'female', 'traveler'],
    'bicycle': ['bike'],
    'car': ['van', 'truck'],
    'motorcycle': ['moped', 'motorcycle'],
    'airplane': ['airplane', 'jet'],
    'bus': ['bus'],
    'train': ['train'],
    'truck': ['truck'],
    'boat': ['boat', 'ship'],
    'traffic light': ['signal'],
    'fire hydrant': ['hydrant'],
    'stop sign': ['stop'],
    'parking meter': ['meter'],
    'bench': ['bench', 'benches'],
    'bird': ['birds'],
    'cat': ['cat', 'cats'],
    'dog': ['dog', 'puppy', 'dogs'],
    'horse': ['horse', 'horses'],
    'sheep': ['sheep'],
    'cow': ['cow', 'cows'],
    'elephant': ['elephant', 'elephants'],
    'bear': ['bear', 'bears'],
    'zebra': ['zebra', 'zebras'],
    'giraffe': ['giraffe', 'giraffes'],
    'backpack': ['backpack'],
    'umbrella': ['umbrella', 'umbrellas'],
    'handbag': ['bag'],
    'tie': ['tie'],
    'suitcase': ['suitcase'],
    'frisbee': ['frisbee'],
    'skis': ['skis'],
    'snowboard': ['snowboard'],
    'sports ball': ['ball'],
    'kite': ['kite', 'kites'],
    'baseball bat': ['bat'],
    'baseball glove': ['glove'],
    'skateboard': ['skateboard'],
    'surfboard': ['surfboard', 'surfboards'],
    'tennis racket': ['racket', 'rackets'],
    'bottle': ['bottle', 'bottles'],
    'wine glass': ['wine glass'],
    'cup': ['cup'],
    'fork': ['fork'],
    'knife': ['knife'],
    'spoon': ['spoon'],
    'bowl': ['bowl'],
    'banana': ['banana', 'bananas'],
    'apple': ['apples'],
    'sandwich': ['sandwich', 'sandwiches'],
    'orange': ['oranges'],
    'broccoli': ['broccoli'],
    'carrot': ['carrot', 'carrots'],
    'hot dog': ['hotdog', 'hotdogs'],
    'pizza': ['pizza', 'pizzas'],
    'donut': ['donut', 'donuts'],
    'cake': ['cake'],
    'chair': ['chair', 'chairs'],
    'couch': ['couch'],
    'potted plant': ['plant'],
    'bed': ['bed'],
    'dining table': ['table', 'desk'],
    'toilet': ['toilet', 'toilets'],
    'tv': ['television', 'television'],
    'laptop': ['laptop'],
    'mouse': ['mouse'],
    'remote': ['remote'],
    'keyboard': ['keyboard'],
    'cell phone': ['cellphone', 'cellphones'],
    'microwave': ['microwave'],
    'oven': ['oven', 'stove'],
    'toaster': ['toaster'],
    'sink': ['sink'],
    'refrigerator': ['refrigerator'],
    'book': ['book'],
    'clock': ['clock'],
    'vase': ['vase', 'vases'],
    'scissors': ['scissors'],
    'teddy bear': ['teddy bear'],
    'hair drier': ['hairdryer'],
    'toothbrush': ['toothbrush'],
}


coco_double_words = [
    'motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal',
    'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball',
    'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone',
    'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer',
    'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track'
]

# Reverse the dictionary to map each synonym to its category
synonym_to_category_reversed = {synonym: category for category, synonyms in synonym_to_category.items() for synonym in synonyms}
synonym_to_category_reversed.update({category: category for category in synonym_to_category.keys()})

def map_object_to_category(obj):
    obj = obj.lower().strip()
    # Check if the object matches any double word in the list
    for double_word in coco_double_words:
        if obj.endswith(double_word):
            return synonym_to_category_reversed.get(double_word, obj)
    # Otherwise, get the last word
    last_word = obj.split()[-1]
    return synonym_to_category_reversed.get(last_word, obj)

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def convert_to_coco_format(x1, y1, x2, y2):
    """
    Convert bounding box coordinates from [x1, y1, x2, y2] to [x, y, width, height].
    """
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1
    return [x, y, width, height]    

def extract_objects_and_bboxes(captions_file):
    """
    Extracts objects and their bounding boxes from a JSON file, mapping objects to the best category,
    and skips non-MSCOCO objects. Each bounding box is listed separately with the object.

    Parameters:
        captions_file (str): Path to the input JSON file.

    Returns:
        list: A list of dictionaries with image filenames and lists of objects with bounding boxes as strings.
    """
    captions_data = load_json(captions_file)
    
    # Initialize a list to store the results
    results = []
    
    # Set of valid MSCOCO object categories
    valid_mscoco_objects = set(synonym_to_category.keys())

    # Process each entry in the input JSON
    for entry in captions_data:
        image_id = entry['image_id']
        caption = entry['caption']
        
        # Extract objects and bounding boxes using updated regex
        matches = re.findall(r'([\w\s]+?) \[\[([\d,;]+)\]\]', caption)
        
        # Initialize a list to store the objects and bounding boxes for this image
        objects_and_bboxes = []
        
        for match in matches:
            obj = match[0].strip()  # Get the object description
            # Check for double words first
            double_word_match = next((dw for dw in coco_double_words if obj.endswith(dw)), None)
            if double_word_match:
                mapped_obj = synonym_to_category_reversed.get(double_word_match, double_word_match)
            else:
                # Otherwise, map using the last word
                last_word = obj.split()[-1]
                mapped_obj = synonym_to_category_reversed.get(last_word, last_word)
            
            # Check if the mapped object is a valid MSCOCO object
            if mapped_obj in valid_mscoco_objects:
                bbox = match[1]
                # Handle multiple bounding boxes
                bbox_list = bbox.split(';')
                for single_bbox in bbox_list:
                    coords = list(map(int, single_bbox.split(',')))
                    coco_bbox = convert_to_coco_format(*coords)                    
                    objects_and_bboxes.append(f"{mapped_obj} {coco_bbox}")
        
        # Add the extracted information to the results list
        results.append({
            "image": f"{image_id}",
            "bbx with object": objects_and_bboxes
        })
    
    return results


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

def save_processed_data(processed_data, output_file):
    with open(output_file, 'w') as file:
        json.dump(processed_data, file, indent=4)


captions_file = '../data/captions/captions.json'
# objects_file = '../data/objects/objects.json'
output_file = '../data/bboxes_objects/synonyms/bboxes_objects.json'


processed_data = extract_objects_and_bboxes(captions_file)
save_processed_data(processed_data, output_file)

