import json

# Define the caption file path
caption_file_path = '../data/captions/captions.json'

# Define the ground truth file path (example file path, adjust as needed)
ground_truth_file_path = '../data/gt_bboxes/gt_bboxes.json'

# Synonym to category mapping
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

# List of double words
coco_double_words = [
    'motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal',
    'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball',
    'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone',
    'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer',
    'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track'
]

# Process captions to extract mentioned objects and sentences
def extract_objects_and_sentences(captions_file_path, synonym_to_category, coco_double_words):
    extracted_data = []
    with open(captions_file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            image_id = item['image_id']
            caption = item['caption']
            objects_in_caption = []
            
            # Normalize and extract objects from the caption
            caption = caption.lower()
            for double_word in coco_double_words:
                caption = caption.replace(double_word, double_word.replace(' ', '_'))

            words = caption.split()
            for word in words:
                word = word.strip('.,!?')
                for category, synonyms in synonym_to_category.items():
                    if word in synonyms:
                        objects_in_caption.append(category)
                        break
            
            extracted_data.append({
                'image_id': image_id,
                'objects': objects_in_caption,
                'caption': caption
            })
    
    return extracted_data

# Extracted data from captions
extracted_data = extract_objects_and_sentences(caption_file_path, synonym_to_category, coco_double_words)

# Process ground truth data to extract objects
def extract_ground_truth_objects(ground_truth_file_path):
    ground_truth_objects = {}
    with open(ground_truth_file_path, 'r') as f:
        ground_truth_data = json.load(f)
        for item in ground_truth_data:
            image_id = item['image']
            objects = []
            for obj in item['bbx with object']:
                obj_name = obj.split('[')[0].strip()
                objects.append(obj_name)
            
            ground_truth_objects[image_id] = objects
    
    return ground_truth_objects

# Extracted ground truth objects
ground_truth_objects = extract_ground_truth_objects(ground_truth_file_path)

# Calculate metrics
def calculate_metrics(extracted_data, ground_truth_objects):
    hallucinated_objects = 0
    total_objects_mentioned = 0
    sentences_with_hallucinated_object = 0
    total_sentences = len(extracted_data)

    for data in extracted_data:
        image_id = data['image_id']
        mentioned_objects = data['objects']
        total_objects_mentioned += len(mentioned_objects)

        if image_id in ground_truth_objects:
            ground_truth = ground_truth_objects[image_id]
            hallucinated_in_sentence = False

            for obj in mentioned_objects:
                if obj not in ground_truth:
                    hallucinated_objects += 1
                    hallucinated_in_sentence = True
            
            if hallucinated_in_sentence:
                sentences_with_hallucinated_object += 1

    CHAIR_i = hallucinated_objects / total_objects_mentioned if total_objects_mentioned else 0
    CHAIR_s = sentences_with_hallucinated_object / total_sentences if total_sentences else 0

    return CHAIR_i, CHAIR_s

# Calculate metrics
CHAIR_i, CHAIR_s = calculate_metrics(extracted_data, ground_truth_objects)

print("CHAIR_i:", CHAIR_i)
print("CHAIR_s:", CHAIR_s)
