import torch
from sentence_transformers import SentenceTransformer
import json

cat_2017 = '../data/annotations/instances_val2017.json'

# get coco classes
json_file = cat_2017
if json_file is not None:
    with open(json_file, 'r') as COCO:
        js = json.loads(COCO.read())
        # Extract and print only the 'name' field from each category
        class_list_coco = [category['name'] for category in js['categories']]


# synonyms = {
#     'person': ['girl', 'boy', 'man', 'woman', 'kid', 'child', 'child', 'chef', 'baker', 'people', 'adult', 'rider', 'children', 'baby', 'worker', 'passenger', 'sister', 'biker', 'policeman', 'cop', 'officer', 'lady', 'cowboy', 'bride', 'groom', 'male', 'female', 'guy', 'traveler', 'mother', 'father', 'gentleman', 'pitcher', 'player', 'players', 'skier', 'skiers', 'snowboarder', 'skater', 'skateboarder', 'foreigner', 'caller', 'offender', 'coworker', 'trespasser', 'patient', 'politician', 'soldier', 'grandchild', 'serviceman', 'walker', 'drinker', 'doctor', 'bicyclist', 'thief', 'buyer', 'teenager', 'student', 'camper', 'driver', 'solider', 'hunter', 'shopper', 'villager', 'friends'],
#     'bicycle': ['bike', 'unicycle', 'minibike', 'trike'],
#     'car': ['automobile', 'van', 'minivan', 'sedan', 'suv', 'hatchback', 'cab', 'jeep', 'coupe', 'taxicab', 'limo', 'taxi'],
#     'motorcycle': ['scooter', 'motor bike', 'motor cycle', 'motorbike', 'moped'],
#     'airplane': ['jetliner', 'plane', 'air plane', 'monoplane', 'aircraft', 'jet', 'airbus', 'biplane', 'seaplane'],
#     'bus': ['minibus', 'trolley', 'buses'],
#     'train': ['locomotive', 'tramway', 'caboose'],
#     'truck': ['pickup', 'lorry', 'hauler', 'firetruck'],
#     'boat': ['boats', 'ship', 'liner', 'sailboat', 'motorboat', 'dinghy', 'powerboat', 'speedboat', 'canoe', 'skiff', 'yacht', 'kayak', 'catamaran', 'pontoon', 'houseboat', 'vessel', 'rowboat', 'trawler', 'ferryboat', 'watercraft', 'tugboat', 'schooner', 'barge', 'ferry', 'sailboard', 'paddleboat', 'lifeboat', 'freighter', 'steamboat', 'riverboat', 'battleship', 'steamship'],
#     'traffic light': ['street light', 'traffic signal', 'stop light', 'streetlight', 'stoplight'],
#     'fire hydrant': ['hydrant'],
#     'bench': ['pew'],
#     'bird': ['ostrich', 'owl', 'seagull', 'goose', 'duck', 'parakeet', 'falcon', 'robin', 'pelican', 'waterfowl', 'heron', 'hummingbird', 'mallard', 'finch', 'pigeon', 'sparrow', 'seabird', 'osprey', 'blackbird', 'fowl', 'shorebird', 'woodpecker', 'egret', 'chickadee', 'quail', 'bluebird', 'kingfisher', 'buzzard', 'willet', 'gull', 'swan', 'bluejay', 'flamingo', 'cormorant', 'parrot', 'loon', 'gosling', 'waterbird', 'pheasant', 'rooster', 'sandpiper', 'crow', 'raven', 'turkey', 'oriole', 'cowbird', 'warbler', 'magpie', 'peacock', 'cockatiel', 'lorikeet', 'puffin', 'vulture', 'condor', 'macaw', 'peafowl', 'cockatoo', 'songbird'],
#     'cat': ['kitten', 'feline', 'tabby'],
#     'dog': ['puppy', 'beagle', 'pup', 'chihuahua', 'schnauzer', 'dachshund', 'rottweiler', 'canine', 'pitbull', 'collie', 'pug', 'terrier', 'poodle', 'labrador', 'doggie', 'doberman', 'mutt', 'doggy', 'spaniel', 'bulldog', 'sheepdog', 'weimaraner', 'corgi', 'cocker', 'greyhound', 'retriever', 'brindle', 'hound', 'whippet', 'husky'],
#     'horse': ['colt', 'pony', 'racehorse', 'stallion', 'equine', 'mare', 'foal', 'palomino', 'mustang', 'clydesdale', 'bronc', 'bronco'],
#     'sheep': ['lamb', 'ram', 'goat', 'ewe'],
#     'cow': ['cattle', 'oxen', 'ox', 'calf', 'holstein', 'heifer', 'buffalo', 'bull', 'zebu', 'bison', 'cows'],
#     'bear': ['panda'],
#     'zebra': ['zebras'],
#     'giraffe': ['giraffes'],
#     'backpack': ['knapsack'],
#     'handbag': ['wallet', 'purse', 'briefcase'],
#     'tie': ['bow', 'bow tie'],
#     'suitcase': ['suit case', 'luggage'],
#     'skis': ['ski'],
#     'sports ball': ['ball'],
#     'surfboard': ['longboard', 'skimboard', 'shortboard', 'wakeboard'],
#     'tennis racket': ['racket'],
#     'bottle': ['bottles'],
#     'knife': ['pocketknife', 'knive'],
#     'bowl': ['container'],
#     'sandwich': ['burger', 'sub', 'cheeseburger', 'hamburger'],
#     'donut': ['doughnut', 'bagel'],
#     'cake': ['cheesecake', 'cupcake', 'shortcake', 'coffeecake', 'pancake'],
#     'chair': ['seat', 'stool'],
#     'couch': ['sofa', 'recliner', 'futon', 'loveseat', 'settee', 'chesterfield'],
#     'potted plant': ['houseplant'],
#     'dining table': ['table', 'desk'],
#     'toilet': ['urinal', 'commode', 'lavatory', 'potty'],
#     'tv': ['monitor', 'televison', 'television'],
#     'laptop': ['computer', 'notebook', 'netbook', 'lenovo', 'macbook', 'laptop computer'],
#     'cell phone': ['mobile phone', 'phone', 'cellphone', 'telephone', 'phon', 'smartphone', 'iPhone'],
#     'oven': ['stovetop', 'stove'],
#     'refrigerator': ['fridge', 'freezer'],
#     'teddy bear': ['teddybear'],
#     'hair drier': ['hairdryer'],
# }
# # Flatten synonyms into a single lookup dictionary
# synonym_to_class = {syn: cls for cls, syns in synonyms.items() for syn in syns}

# get objects aka the classes that cogvlm predicts
unique_objects = set()
with open('../data/bboxes_objects/bboxes_original_objects.json', 'r') as file:
    data = json.load(file)
    for entry in data:
        for bbx in entry['bbx with object']:
            object_name = bbx.split(' [')[0]
            unique_objects.add(object_name)

final_class_list = list(unique_objects)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text):
    output = model.encode(text)
    return torch.tensor(output)

gt_classes = class_list_coco
predictions = final_class_list

# Generate embeddings only for the actual COCO classes and synonyms
gt_embeddings = {cls: get_embedding(cls) for cls in gt_classes}
# synonym_embeddings = {syn: get_embedding(syn) for syn in synonym_to_class}
prediction_embeddings = {pred: get_embedding(pred) for pred in predictions}

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0)

threshold1 = 0.6  # Adjust based on desired strictness
mapped_predictions1 = {}

for pred, pred_embedding in prediction_embeddings.items():
    # Check similarity with COCO classes first
    similarities = {gt: cosine_similarity(pred_embedding, gt_embeddings[gt]) for gt in gt_classes}
    best_match, best_sim = max(similarities.items(), key=lambda x: x[1])
    
    # If no good match found, check with synonyms
    # if best_sim < threshold1:
    #     similarities = {syn: cosine_similarity(pred_embedding, synonym_embeddings[syn]) for syn in synonym_to_class}
    #     best_match, best_sim = max(similarities.items(), key=lambda x: x[1])
    #     if best_sim > threshold1:
    #         best_match = synonym_to_class[best_match]

    if best_sim > threshold1:
        if best_match in mapped_predictions1:
            mapped_predictions1[best_match].append(pred)
        else:
            mapped_predictions1[best_match] = [pred]

with open('../data/classes/object_to_class_0.6.json', 'w') as file:
    json.dump(mapped_predictions1, file, indent=4)


# ['sheets', 'girls', 'bag', 'painting', 'men', 'sconces', 'desk', 'trunk', 'surfboards', 'beard', 'water', 'tent', 'bun', 'cabinets', 
# 'donut', 'hat', 'raincoat', 'clock', 'belt', 'food', 'toothbrush', 'chandelier', 'book', 'ctr', 'monitors', 'pitcher', 'game', 'lodge', 
# 'apples', 'bride', 'sandwich', 'oven', 'bucket', 'window', 'screen', 'fruit', 'chips', 'cheese', 'oranges', 'man', 'stove', 'kite', 
# 'women', 'railing', 'surface', 'toilet', 'scissors', 'head', 'florets', 'person', 'pan', 'Frisbee', 'guitar', 'backpack', 'stone', 'sink', 
# 'racket', 'microwave', 'fence', 'baby', 'street', 'cellphone', 'tie', 'mound', 'poles', 'poodle', 'catcher', 'pancakes', 'ribbon', 'statues', 
# 'scaffold', 'robe', 'jacket', 'strawberries', 'hotdog', 'handle', 'mountains', 'pen', 'bed', 'pink', 'stump', 'roses', 'Graffiti', 'control', 
# 'lids', 'uniforms', 'sugar', 'vanity', 'field', 'crowd', 'cake', 'cat', 'tomato', 'mirror', 'snowboard', 'building', 'zebras', 'shelf', 
# 'lake', 'drawers', '2', 'photos', 'bears', 'stop', 'sweatshirt', 'people', 'nose', 'snow', 'bedspread', 'light', 'ramp', 'dress', 'remote', 
# 'flag', 'calf', 'wagon', 'flags', 'hands', 'cilantro', 'airplane', 'fixture', 'coach', 'platform', 'boy', 'nuts', 'camera', 'puppy', 'runway', 
# 'signal', 'pants', 'tureen', 'toddler', 'rackets', 'meters', 'tree', 'lid', 'People', 'syrup', 'road', 'giraffes', 'backsplash', 'meter', 
# 'sauce', 'skateboarder', 'clocks', 'couch', 'smoke', 'title', 'countertop', 'refrigerator', 'broccoli', 'beer', 'restaurant', 'boat', 'top', 
# 'sheep', 'train', 'hill', 'pieces', 'wine', 'banana', 'bread', 'tablecloth', 'carrots', 'bowl', 'headlight', 'spots', 'bridge', 'stripes', 
# 'sandwiches', 'soccer', 'children', 'branch', 'jets', 'bar', 'pizzas', 'station', 'bushes', 'zebra', 'skier', 'umbrellas', 'rock', 'cardboard', 
# 'players', 'outfits', 'flowers', 'bottles', 'ledge', 'wrap', 'bakery', 'lettuce', 'Skiers', 'garden', 'kitchen', 'tulips', 'boots', 'tray', 
# 'hair', 'signs', 'helmet', 'newspaper', 'bathtub', 'church', 'rail', 'sprinkles', 'passengers', 'candle', 'stool', 'beach', 'bikinis', 'brush', 
# 'shoulder', 'skirt', 'rocks', 'chair', 'curb', 'ship', 'cupcake', 'surfer', 'motorcycle', 'animal', 'porch', 'ball', 'drinks', 'hand', 'tracks', 
# 'shuttle', 'groom', 'door', 'design', 'cats', 'eyes', 'woman', 'face', 'magazine', 'papers', 'cream', 'boys', 'tea', 'cemetery', 'faces', 
# 'hotdogs', 'wall', 'shoes', 'suitcase', 'pitch', 'faucet', 'kites', 'History', 'feet', 'plant', 'frisbees', 'factory', 'daisies', 'cushion', 
# 'bear', 'canal', 'table', 'step', 'juice', 'elephants', 'waffle', 'base', 'river', 'lot', 'cookies', 'ocean', 'umbrella', 'lounge', 'cup', 
# 'shorts', 'bull', 'comforter', 'mouth', 'jersey', 'plane', '12', 'lamp', 'butter', 'team', 'family', 'expression', 'vase', 'shirts', 'Elephants', 
# 'jeans', 'armor', 'cabinet', 'words', 'button', 'van', 'kitten', 'slices', 'cover', 'counter', 'seat', 'uniform', 'disc', 'store', 'surfboard', 
# 'leaves', 'vases', 'cloth', 'case', 'containers', 'suits', 'tire', 'cheesecake', 'buses', 'bird', 'skis', 'boats', 'rooster', 'house', 'sidewalk', 
# 'teammates', 'chairs', 'grass', 'trees', 'sky', 'pole', 'luggage', 'pears', 'spires', 'baseman', 'mouse', 'laptop', 'baseball', 'berries', 
# 'horses', 'cellphones', 'ambulance', 'skateboard', 'horse', 'bat', 'outfit', 'cable', 'dish', 'player', 'graffiti', 'coffee', 'parade', 
# 'shirt', 'donuts', 'computer', 'shore', 'wallpaper', 'meat', 'blanket', 'floor', 'keyboard', 'stoves', 'harbor', 'bike', 'white', 'toilets', 
# 'child', 'pancake', 'bottle', 'sweater', 'plate', 'pavement', 'fountain', '"Fun', '40', 'Sailors', 'creamer', 'dogs', 'paws', 'Golf', 'word', 
# 'motorcycles', 'fish', 'wave', 'spoon', 'giraffe', 'car', 'icing', 'wires', 'cow', 'frisbee', 'plaque', "'Gulf", 'cows', 'dog', 'deer', 'meal', 
# 'hydrant', 'truck', 'moose', 'suit', 'bananas', 'chicken', 'items', 'tower', 'pizza', 'skiers', 'phone', 'turban', 'bow', 'spear', 'bicycle', 
# 'area', 'forks', 'balloon', 'sign', 'mountain', 'bench', 'benches', 'stripe', 'beak', 'path', 'bus', 'lighter', 'girl', 'moped', 'coat', 
# 'potatoes', 'sailors', 'doll', 'elephant', 'television', 'doughnuts', 'back', 'glasses', 'box', 'fork', 'bagel', 'fireplace', 'blender', 
# 'friends', 'moon']