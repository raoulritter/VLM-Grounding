import json
import torch
from torchvision.ops import box_iou

file_path = 'VLM-Grounding/COGVLM/data/bbxes_objects/bbxes_objects.json'


with open(file_path, 'r') as file:
    data = json.load(file)

extracted_data = []

for item in data:
    image_id = str(int(item['image'].split('.')[0]))  # Convert to int and back to str to remove leading zeros
    bboxes = []
    labels = []
    if "bbx with object" in item:
        for bbx in item["bbx with object"]:
            if bbx != []:
                # Extract the bounding box coordinates and label
                label = bbx.split(' [')[0]  # Split and take the first part as label
                coords = bbx[bbx.index('[') + 1:bbx.index(']')].split(', ')
                # Convert string coordinates to integers
                coords = list(map(int, coords))
                # Append the bounding box to the list
                bboxes.append(coords)
                labels.append(label)
    extracted_data.append({"image_id": image_id, "bboxes": bboxes, "labels": labels})
predicted_bboxes = extracted_data

with open('VLM-Grounding/COGVLM/data/gt_bboxes/gt_bboxes.json', 'r') as file:
    gt_bboxes = [json.loads(line) for line in file]


# This script processes pairs of predicted and ground truth bounding boxes for object detection. 
# It calculates the Intersection over Union (IoU) for each predicted box against all ground truth boxes in the same image, 
# identifies the ground truth box with the highest IoU for each prediction, and classifies the prediction based on the IoU value. 
# Predictions with an IoU below 0.5 are considered hallucinations, while those with higher IoU are noted for further evaluation. 
hallucinations = []
correct_classifications = []
misclassifications = []
classification = []
for pred_bbx in predicted_bboxes:
    for gt_bbx in gt_bboxes:
        # Check if the current predicted and ground truth boxes are from the same image
        if pred_bbx['image_id'] == gt_bbx['image_id']:
            pred_boxes = torch.tensor(pred_bbx['bboxes'], dtype=torch.float32)
            gt_boxes = torch.tensor(gt_bbx['bboxes'], dtype=torch.float32)

            # some tensors were empty
            if pred_boxes.nelement() == 0 or gt_boxes.nelement() == 0:
                print("One of the tensors is empty.")
                continue

            current_iou = box_iou(pred_boxes, gt_boxes)
            # Find the maximum IoU value for each predicted box and the index of the corresponding ground truth box
            max_iou_values, max_indices = current_iou.max(dim=1)

            # Iterate over each predicted box to classify based on the IoU value
            for idx, max_iou in enumerate(max_iou_values):
                # Get the index of the ground truth box that has the highest IoU with the current predicted box
                matched_gt_idx = max_indices[idx]
                # Retrieve the coordinates of the matched ground truth box
                matched_gt_box = gt_bbx['bboxes'][matched_gt_idx]
                # Retrieve the coordinates of the current predicted box
                pred_box = pred_bbx['bboxes'][idx]

                if max_iou < 0.5:
                    # If IoU is less than 0.5, classify as a hallucination
                    hallucinations.append({"image_id": pred_bbx['image_id'], "pred_box": pred_box, "iou": max_iou.item()})
                else:                
                    classification.append({"image_id": pred_bbx['image_id'], "pred_box": pred_box, "iou": max_iou.item()})



print(f"Hallucinations: {len(hallucinations)}")
print(f"Classification: {len(classification)}")
print(classification)



