# Modified script to handle maximum IoU values and classify correctly, misclassified, and hallucinations

import json
from typing import List, Dict

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = w1 * h1
    box2Area = w2 * h2

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

# Function to parse bounding box from string
def parse_bounding_box(bbox_str):
    bbox_str = bbox_str.split('[')[-1].split(']')[0]  # Get the part inside the square brackets
    bbox = list(map(float, bbox_str.split(',')))
    return bbox

# Load JSON files
with open('../data/bbxes_objects/adjusted_bbox_objects.json', 'r') as f:
    predicted_data = json.load(f)

with open('../data/gt_bboxes/gt_bboxes.json', 'r') as f:
    ground_truth_data = json.load(f)

hallucinations = []
correctly_classified = []
misclassified = []
instances = 0

# Process each image and calculate IoU
for pred in predicted_data:
    image_id = pred['image']
    pred_boxes = pred['bbx with object']
    instances += len(pred_boxes)

    # Find the corresponding ground truth boxes
    gt = next((item for item in ground_truth_data if item['image'] == image_id), None)
    if gt is not None:
        gt_boxes = gt['bbx with object']
        for pred_box in pred_boxes:
            pred_box_parsed = parse_bounding_box(pred_box)
            max_iou = 0
            matched_gt_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                gt_box_parsed = parse_bounding_box(gt_box)
                iou = calculate_iou(pred_box_parsed, gt_box_parsed)
                if iou > max_iou:
                    max_iou = iou
                    matched_gt_idx = idx
            
            pred_label = pred_box.split('[')[0].strip()
            matched_gt_label = gt_boxes[matched_gt_idx].split('[')[0].strip() if matched_gt_idx >= 0 else None

            if max_iou < 0.5:
                # If IoU is less than 0.5, classify as a hallucination
                hallucinations.append({"image_id": image_id, "pred_box": pred_box, "iou": max_iou})
            else:
                # Check if the labels match
                if matched_gt_label == pred_label:
                    correctly_classified.append({"image_id": image_id, "pred_box": pred_box, "iou": max_iou, "label": pred_label})
                else:
                    misclassified.append({"image_id": image_id, "pred_box": pred_box, "iou": max_iou, "pred_label": pred_label, "gt_label": matched_gt_label})

print(f"Total instances: {instances}")
print(f"Number of hallucinations: {len(hallucinations)}")
print(f"Number of correct classifications: {len(correctly_classified)}")
print(f"Number of misclassifications: {len(misclassified)}")

# Save the results to JSON files
with open('../data/output/hallucinations.json', 'w') as f:
    json.dump(hallucinations, f, indent=4)

with open('../data/output/correctly_classified.json', 'w') as f:
    json.dump(correctly_classified, f, indent=4)

with open('../data/output/misclassified.json', 'w') as f:
    json.dump(misclassified, f, indent=4)
