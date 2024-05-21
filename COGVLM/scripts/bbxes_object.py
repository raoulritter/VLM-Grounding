import json
import re

# Create a dictionary mapping synonyms to categories
synonym_to_category = {
    'person': ['girl', 'boy', 'man', 'woman', 'kid', 'child', 'chef', 'baker', 'people', 'adult', 'rider', 'children', 'baby', 'worker', 'passenger', 'sister', 'biker', 'policeman', 'cop', 'officer', 'lady', 'cowboy', 'bride', 'groom', 'male', 'female', 'guy', 'traveler', 'mother', 'father', 'gentleman', 'pitcher', 'player', 'skier', 'snowboarder', 'skater', 'skateboarder', 'foreigner', 'caller', 'offender', 'coworker', 'trespasser', 'patient', 'politician', 'soldier', 'grandchild', 'serviceman', 'walker', 'drinker', 'doctor', 'bicyclist', 'thief', 'buyer', 'teenager', 'student', 'camper', 'driver', 'solider', 'hunter', 'shopper', 'villager'],
    'bicycle': ['bike', 'unicycle', 'minibike', 'trike'],
    'car': ['automobile', 'van', 'minivan', 'sedan', 'suv', 'hatchback', 'cab', 'jeep', 'coupe', 'taxicab', 'limo', 'taxi'],
    'motorcycle': ['scooter', 'motor bike', 'motor cycle', 'motorbike', 'moped'],
    'airplane': ['jetliner', 'plane', 'air plane', 'monoplane', 'aircraft', 'jet', 'airbus', 'biplane', 'seaplane'],
    'bus': ['minibus', 'trolley'],
    'train': ['locomotive', 'tramway', 'caboose'],
    'truck': ['pickup', 'lorry', 'hauler', 'firetruck'],
    'boat': ['ship', 'liner', 'sailboat', 'motorboat', 'dinghy', 'powerboat', 'speedboat', 'canoe', 'skiff', 'yacht', 'kayak', 'catamaran', 'pontoon', 'houseboat', 'vessel', 'rowboat', 'trawler', 'ferryboat', 'watercraft', 'tugboat', 'schooner', 'barge', 'ferry', 'sailboard', 'paddleboat', 'lifeboat', 'freighter', 'steamboat', 'riverboat', 'battleship', 'steamship'],
    'traffic light': ['street light', 'traffic signal', 'stop light', 'streetlight', 'stoplight'],
    'fire hydrant': ['hydrant'],
    'stop sign': [],
    'parking meter': [],
    'bench': ['pew'],
    'bird': ['ostrich', 'owl', 'seagull', 'goose', 'duck', 'parakeet', 'falcon', 'robin', 'pelican', 'waterfowl', 'heron', 'hummingbird', 'mallard', 'finch', 'pigeon', 'sparrow', 'seabird', 'osprey', 'blackbird', 'fowl', 'shorebird', 'woodpecker', 'egret', 'chickadee', 'quail', 'bluebird', 'kingfisher', 'buzzard', 'willet', 'gull', 'swan', 'bluejay', 'flamingo', 'cormorant', 'parrot', 'loon', 'gosling', 'waterbird', 'pheasant', 'rooster', 'sandpiper', 'crow', 'raven', 'turkey', 'oriole', 'cowbird', 'warbler', 'magpie', 'peacock', 'cockatiel', 'lorikeet', 'puffin', 'vulture', 'condor', 'macaw', 'peafowl', 'cockatoo', 'songbird'],
    'cat': ['kitten', 'feline', 'tabby'],
    'dog': ['puppy', 'beagle', 'pup', 'chihuahua', 'schnauzer', 'dachshund', 'rottweiler', 'canine', 'pitbull', 'collie', 'pug', 'terrier', 'poodle', 'labrador', 'doggie', 'doberman', 'mutt', 'doggy', 'spaniel', 'bulldog', 'sheepdog', 'weimaraner', 'corgi', 'cocker', 'greyhound', 'retriever', 'brindle', 'hound', 'whippet', 'husky'],
    'horse': ['colt', 'pony', 'racehorse', 'stallion', 'equine', 'mare', 'foal', 'palomino', 'mustang', 'clydesdale', 'bronc', 'bronco'],
    'sheep': ['lamb', 'ram', 'goat', 'ewe'],
    'cow': ['cattle', 'oxen', 'ox', 'calf', 'holstein', 'heifer', 'buffalo', 'bull', 'zebu', 'bison'],
    'elephant': [],
    'bear': ['panda'],
    'zebra': [],
    'giraffe': [],
    'backpack': ['knapsack'],
    'umbrella': [],
    'handbag': ['wallet', 'purse', 'briefcase'],
    'tie': ['bow', 'bow tie'],
    'suitcase': ['suit case', 'luggage'],
    'frisbee': [],
    'skis': ['ski'],
    'snowboard': [],
    'sports ball': ['ball'],
    'kite': [],
    'baseball bat': [],
    'baseball glove': [],
    'skateboard': [],
    'surfboard': ['longboard', 'skimboard', 'shortboard', 'wakeboard'],
    'tennis racket': ['racket'],
    'bottle': [],
    'wine glass': [],
    'cup': [],
    'fork': [],
    'knife': ['pocketknife', 'knive'],
    'spoon': [],
    'bowl': ['container'],
    'banana': [],
    'apple': [],
    'sandwich': ['burger', 'sub', 'cheeseburger', 'hamburger'],
    'orange': [],
    'broccoli': [],
    'carrot': [],
    'hot dog': [],
    'pizza': [],
    'donut': ['doughnut', 'bagel'],
    'cake': ['cheesecake', 'cupcake', 'shortcake', 'coffeecake', 'pancake'],
    'chair': ['seat', 'stool'],
    'couch': ['sofa', 'recliner', 'futon', 'loveseat', 'settee', 'chesterfield'],
    'potted plant': ['houseplant'],
    'bed': [],
    'dining table': ['table', 'desk'],
    'toilet': ['urinal', 'commode', 'lavatory', 'potty'],
    'tv': ['monitor', 'televison', 'television'],
    'laptop': ['computer', 'notebook', 'netbook', 'lenovo', 'macbook', 'laptop computer'],
    'mouse': [],
    'remote': [],
    'keyboard': [],
    'cell phone': ['mobile phone', 'phone', 'cellphone', 'telephone', 'phon', 'smartphone', 'iPhone'],
    'microwave': [],
    'oven': ['stovetop', 'stove', 'stove top oven'],
    'toaster': [],
    'sink': [],
    'refrigerator': ['fridge', 'freezer'],
    'book': [],
    'clock': [],
    'vase': [],
    'scissors': [],
    'teddy bear': ['teddybear'],
    'hair drier': ['hairdryer'],
    'toothbrush': []
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

def extract_objects_and_bboxes(input_json_path, output_json_path):
    """
    Extracts objects and their bounding boxes from a JSON file, mapping objects to the best category.

    Parameters:
        input_json_path (str): Path to the input JSON file.

    Returns:
        dict: A dictionary with image IDs as keys and lists of objects and bounding boxes as values.
    """
    # Read the input JSON file
    with open(input_json_path, 'r') as file:
        data = json.load(file)
    
    # Initialize a dictionary to store the results
    result = {}
    
    # Process each entry in the input JSON
    for entry in data:
        image_id = entry['image_id']
        caption = entry['caption']
        
        # Extract objects and bounding boxes using regex
        matches = re.findall(r'([\w\s]+?) \[\[(\d+,\d+,\d+,\d+)\]\]', caption)
        
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
            bbox = list(map(int, match[1].split(',')))
            objects_and_bboxes.append({'object': mapped_obj, 'bbox': bbox})
        
        # Add the extracted information to the result dictionary    
        result[image_id] = objects_and_bboxes
    
    # return result

 # Write the result to the output JSON file
    with open(output_json_path, 'w') as outfile:
        json.dump(result, outfile, indent=4)

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
            bboxes.append([coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1], f"{coords[0]},{coords[1]},{coords[2]},{coords[3]}"])
    return bboxes


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
            combined_data["bbx with object"].append(f"{objects[i]} {bboxes[i][:4]}")

        # Handle cases where there are more bounding boxes than objects
        for i in range(min_length, len(bboxes)):
            bbox_text = bboxes[i][4]
            preceding_word = get_word_preceding_bbox(caption, bbox_text)
            combined_data["bbx with object"].append(f"{preceding_word} {bboxes[i][:4]}")

        processed_data.append(combined_data)

    return processed_data

def save_processed_data(processed_data, output_file):
    with open(output_file, 'w') as file:
        json.dump(processed_data, file, indent=4)


captions_file = '../data/captions/captions.json'
objects_file = '../data/objects/objects.json'
output_file = '../data/bbxes_objects/bbxes_objects.json'

# processed_data = process_entries(captions_file, objects_file)
# save_processed_data(processed_data, output_file)
# print(extract_bboxes(captions_file))
extract_objects_and_bboxes(captions_file, objects_file)

# print(get_word_preceding_bbox(captions_file[0], extract_bboxes(captions_file[0])))