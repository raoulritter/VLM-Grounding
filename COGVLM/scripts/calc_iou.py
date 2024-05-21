import json
import torch
from torchvision.ops import box_iou

# Define file paths
predicted_bboxes_path = 'VLM-Grounding/COGVLM/data/bbxes_objects/adjusted_bbox_objects.json'
gt_bboxes_path = 'VLM-Grounding/COGVLM/data/gt_bboxes/gt_bboxes.json'

# Load the JSON data for predicted bounding boxes
with open(predicted_bboxes_path, 'r') as file:
    predicted_data = json.load(file)

# Load the JSON data for ground truth bounding boxes
with open(gt_bboxes_path, 'r') as file:
    gt_data = json.load(file)

# Extract and adjust the predicted bounding boxes
predicted_bboxes = []
gt_bboxes = []

for item in predicted_data:
    # image_id = str(int(item['image'].split('.')[0]))  # Convert to int and back to str to remove leading zeros
    bboxes = []
    labels = []
    if "bbx with object" in item:
        for bbx in item["bbx with object"]:
            if bbx != []:
                # Extract the bounding box coordinates and label
                label = bbx.split(' [')[0]  # Split and take the first part as label
                coords = bbx[bbx.index('[') + 1:bbx.index(']')].split(', ')
                # Convert string coordinates to floats
                coords = list(map(float, coords))
                # Append the bounding box to the list
                bboxes.append(coords)
                labels.append(label)
    predicted_bboxes.append({"image": item['image'], "bboxes": bboxes, "labels": labels})

for item in gt_data:
    # image_id = str(int(item['image'].split('.')[0]))  # Convert to int and back to str to remove leading zeros
    bboxes = []
    labels = []
    if "bbx with object" in item:
        for bbx in item["bbx with object"]:
            if bbx != []:
                # Extract the bounding box coordinates and label
                label = bbx.split(' [')[0]  # Split and take the first part as label
                coords = bbx[bbx.index('[') + 1:bbx.index(']')].split(', ')
                # Convert string coordinates to floats
                coords = list(map(float, coords))
                # Append the bounding box to the list
                bboxes.append(coords)
                labels.append(label)
    gt_bboxes.append({"image": item['image'], "bboxes": bboxes, "labels": labels})

# print(len(predicted_bboxes))
# Ground truth bounding boxes are already in a usable format
# gt_bboxes = gt_data

# This script processes pairs of predicted and ground truth bounding boxes for object detection. 
# It calculates the Intersection over Union (IoU) for each predicted box against all ground truth boxes in the same image, 
# identifies the ground truth box with the highest IoU for each prediction, and classifies the prediction based on the IoU value. 
# Predictions with an IoU below 0.5 are considered hallucinations, while those with higher IoU are noted for further evaluation. 
hallucinations = []
correctly_classified = []
misclassified = []

# print(len(gt_bboxes))
for pred_bbx in predicted_bboxes:
    for gt_bbx in gt_bboxes:
        # Check if the current predicted and ground truth boxes are from the same image
        if pred_bbx['image'] == gt_bbx['image']:
            pred_boxes = torch.tensor(pred_bbx['bboxes'], dtype=torch.float32)
            gt_boxes = torch.tensor(gt_bbx['bboxes'], dtype=torch.float32)

            # some tensors were empty
            if pred_boxes.nelement() == 0 or gt_boxes.nelement() == 0:
                #print("One of the tensors is empty.")
                continue

            current_iou = box_iou(pred_boxes, gt_boxes)
            # Find the maximum IoU value for each predicted box and the index of the corresponding ground truth box
            max_iou_values, max_indices = current_iou.max(dim=1)

            # Iterate over each predicted box to classify based on the IoU value
            for idx, max_iou in enumerate(max_iou_values):
                # Get the index of the ground truth box that has the highest IoU with the current predicted box
                matched_gt_idx = max_indices[idx]
                # Retrieve the labels of the matched ground truth box and the current predicted box
                matched_gt_label = gt_bbx['labels'][matched_gt_idx]
                pred_label = pred_bbx['labels'][idx]

                if max_iou < 0.5:
                    # If IoU is less than 0.5, classify as a hallucination
                    hallucinations.append({"image_id": pred_bbx['image'], "pred_box": pred_bbx['bboxes'][idx], "iou": max_iou.item()})
                else:
                    # Check if the labels match
                    if matched_gt_label == pred_label:
                        correctly_classified.append({"image_id": pred_bbx['image'], "pred_box": pred_bbx['bboxes'][idx], "iou": max_iou.item(), "label": pred_label})
                    else:
                        misclassified.append({"image_id": pred_bbx['image'], "pred_box": pred_bbx['bboxes'][idx], "iou": max_iou.item(), "pred_label": pred_label, "gt_label": matched_gt_label})

print(f"Hallucinations: {len(hallucinations)}")
print(f"Correctly classified: {len(correctly_classified)}")
print(f"Misclassified: {len(misclassified)}")


with open('VLM-Grounding/COGVLM/data/output/correctly_classified.json', 'w') as file:
    json.dump(correctly_classified, file, indent=4)

with open('VLM-Grounding/COGVLM/data/output/misclassified.json', 'w') as file:
    json.dump(misclassified, file, indent=4)

with open('VLM-Grounding/COGVLM/data/output/hallucinations.json', 'w') as file:
    json.dump(hallucinations, file, indent=4)

