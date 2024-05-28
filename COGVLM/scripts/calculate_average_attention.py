import json
import numpy as np
import torch
import torch.nn.functional as F
import os

# Function to parse bounding box from string
def parse_bounding_box(bbox_str):
    bbox_str = bbox_str.split('[')[-1].split(']')[0]  # Get the part inside the square brackets
    bbox = list(map(float, bbox_str.split(',')))
    return bbox

# Function to find bbox with highest average attention
def find_bbox_with_highest_avg_attention(heatmap, bboxes):
    max_avg_attention = float('-inf')
    best_bbox = None

    # Extract the activations from the heatmap
    activations = heatmap[0, 0, :, :]

    for bbox in bboxes:
        left, top, width, height = map(int, bbox)
        bbox_area = width * height
        attention_sum = activations[top:top+height, left:left+width].sum().item()
        avg_attention = attention_sum / bbox_area

        if avg_attention > max_avg_attention:
            max_avg_attention = avg_attention
            best_bbox = bbox

    return best_bbox, max_avg_attention

# Load JSON files
image_ids_path = '../data/image_ids.json'
ground_truth_path = '../data/gt_bboxes/gt_bboxes.json'

with open(image_ids_path, 'r') as f:
    image_ids = json.load(f)

with open(ground_truth_path, 'r') as f:
    ground_truth_data = json.load(f)

hallucinations = []
correctly_classified = []
misclassified = []
wrong_object = []
instances = 0

# image_ids = ['000000000042.jpg']
# Process each image and calculate attention scores
for image_info in image_ids:
    # image_id = image_info['image_id']
    image_id = image_info
    image_folder = os.path.splitext(image_id)[0]

    # Find the corresponding ground truth boxes
    gt = next((item for item in ground_truth_data if item['image'] == image_id), None)
    # gt = {
    #     "image": "000000000042.jpg",
    #     "bbx with object": [
    #         "dog [214.15, 41.29, 348.26, 243.78]"
    #     ]
    # }
    if gt is None:
        continue

    gt_boxes = [parse_bounding_box(bbox) for bbox in gt['bbx with object']]
    gt_labels = [bbox.split('[')[0].strip() for bbox in gt['bbx with object']]

    # Process heatmaps for each instance
    heatmap_files = [f for f in os.listdir(f"../data/heatmaps/{image_folder}") if f.startswith("explainability_map_") and f.endswith(".pt")]
    for heatmap_file in heatmap_files:
        heatmap_path = os.path.join(f"../data/heatmaps/{image_folder}", heatmap_file)
        
        heatmap = torch.load(heatmap_path, map_location=torch.device('cpu'))
        heatmap = F.interpolate(heatmap, scale_factor=14, mode='bilinear')  # [B, 1, H, W]

        max_gt_bbox, max_attention_score = find_bbox_with_highest_avg_attention(heatmap, gt_boxes)
        matched_gt_idx = gt_boxes.index(max_gt_bbox)
        matched_gt_label = gt_labels[matched_gt_idx]

        # Extract the predicted label from the heatmap file name
        pred_label = heatmap_file.split("explainability_map_")[-1].split(".pt")[0]  # Adjust this to actual predicted label extraction logic
        print(f'{max_attention_score} of {pred_label}')
        if max_attention_score < 0.3:
            if matched_gt_label == pred_label:
                hallucinations.append({"image_id": image_id, "pred_label": pred_label, "score": max_attention_score})
            else: 
                wrong_object.append({"image_id": image_id, "pred_label": pred_label, "score": max_attention_score})
        else:
            if matched_gt_label == pred_label:
                correctly_classified.append({"image_id": image_id, "pred_label": pred_label, "score": max_attention_score})
            else:
                misclassified.append({"image_id": image_id, "pred_label": pred_label, "gt_label": matched_gt_label, "score": max_attention_score})

        instances += 1

# Save the results to JSON files
base_path_output = '../data/average_attention_output'

with open(f'{base_path_output}/hallucinations.json', 'w') as f:
    json.dump(hallucinations, f, indent=4)

with open(f'{base_path_output}/correctly_classified.json', 'w') as f:
    json.dump(correctly_classified, f, indent=4)

with open(f'{base_path_output}/misclassified.json', 'w') as f:
    json.dump(misclassified, f, indent=4)

with open(f'{base_path_output}/wrong_object.json', 'w') as f):
    json.dump(wrong_object, f, indent=4)

# # Calculate average attention scores
def calculate_average_attention(data):
    if not data:
        return 0
    total_attention = sum(item['score'] for item in data)
    return total_attention / len(data)

avg_attention_hallucinations = calculate_average_attention(hallucinations)
avg_attention_correctly_classified = calculate_average_attention(correctly_classified)
avg_attention_misclassified = calculate_average_attention(misclassified)
avg_attention_wrong_objects = calculate_average_attention(wrong_object)

total_attention = hallucinations + correctly_classified + misclassified
avg_attention_total = calculate_average_attention(total_attention)

print(f"Total instances: {instances}")
print(f"Number of hallucinations: {len(hallucinations)}, Average attention: {avg_attention_hallucinations}")
print(f"Number of correct classifications: {len(correctly_classified)}, Average attention: {avg_attention_correctly_classified}")
print(f"Number of misclassifications: {len(misclassified)}, Average attention: {avg_attention_misclassified}")
print(f"Number of wrong objects: {len(wrong_object)}, Average attention: {avg_attention_wrong_objects}")
print(f"Final Average attention (excluding wrong objects): {avg_attention_total} \n")

if (instances - len(wrong_object)) != 0:
    print(f"Percentage of Hallucinations: {len(hallucinations)/(instances-len(wrong_object))}")
    print(f"Percentage of misclassifications: {len(misclassified)/(instances-len(wrong_object))}")
    print(f"Percentage of correct classifications: {len(correctly_classified)/(instances-len(wrong_object))}")
else:
    print("No correct objects found")


