import sys
from nltk.stem import *
import nltk
import json
from pattern.en import singularize
import argparse
from tqdm import tqdm

lemma = nltk.wordnet.WordNetLemmatizer()
synonyms = open('VLM-Grounding/COGVLM/data/synonyms.txt').readlines()
synonyms = [s.strip().split(', ') for s in synonyms]
mscoco_objects = [] #mscoco objects and *all* synonyms
inverse_synonym_dict = {}
for synonym in synonyms:
    mscoco_objects.extend(synonym)
    for s in synonym:
        inverse_synonym_dict[s] = synonym[0]



coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
vehicle_words = ['jet', 'train']

double_word_dict = {}
for double_word in coco_double_words:
    double_word_dict[double_word] = double_word
for animal_word in animal_words:
    double_word_dict['baby %s' %animal_word] = animal_word
    double_word_dict['adult %s' %animal_word] = animal_word
for vehicle_word in vehicle_words:
    double_word_dict['passenger %s' %vehicle_word] = vehicle_word
double_word_dict['bow tie'] = 'tie'
double_word_dict['toilet seat'] = 'toilet'
double_word_dict['wine glas'] = 'wine glass'



#standard preprocessing
with open('VLM-Grounding/COGVLM/data/captions/captions.json', 'r') as file:
    captions = [json.loads(line) for line in file]

objects = [] 

for caption_entry in tqdm(captions, desc="Processing Captions"):
    caption = caption_entry['caption']
    image_filename = caption_entry['image_id']

    words = nltk.word_tokenize(caption.lower())
    words = [singularize(w) for w in words]

    # Filter out unwanted characters and special tokens
    words = [w for w in words if w.isalpha()]

    # Replace double words
    i = 0
    double_words = []
    idxs = []
    while i < len(words):
        idxs.append(i)
        double_word = ' '.join(words[i:i+2])
        if double_word in double_word_dict:
            double_words.append(double_word_dict[double_word])
            i += 2
        else:
            double_words.append(words[i])
            i += 1
    words = double_words

    # Handle special cases
    if 'toilet' in words and 'seat' in words:
        words = [word for word in words if word != 'seat']

    # Get synonyms for all words in the caption
    idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(mscoco_objects)]
    words = [word for word in words if word in set(mscoco_objects)]
    node_words = []
    for word in words:
        node_words.append(inverse_synonym_dict[word])
        
    data = {
        "image": image_filename,
        "objects": node_words
    }
    objects.append(data)


with open('VLM-Grounding/COGVLM/data/objects/objects.json', 'w') as f:
    for object in objects:
        json.dump(object, f)
        f.write('\n')

