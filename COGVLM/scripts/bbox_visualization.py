import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# threshold = 0.7
# Load the JSON data
with open('../data/bboxes_objects/synonyms/adjusted_bboxes_objects.json', 'r') as file:
    predicted_bboxes = json.load(file)

# with open(f'../data/bboxes_objects/map_classes/bboxes_objects_{threshold}.json', 'r') as file:
#     predicted_bboxes = json.load(file)

# with open('../data/bboxes_objects/bboxes_original_removed.json', 'r') as file:
#     predicted_bboxes = json.load(file)

with open('../data/gt_bboxes/gt_bboxes.json', 'r') as file:
    ground_truth_bboxes = json.load(file)

# Define the path to the images folder
images_folder = '../data/images'
output_folder = '../data/bbox_images/synonyms'
# output_folder = f'../data/bbox_images/map_classes/{threshold}'
# output_folder = '../data/bbox_images/skip_classes'

hallucinations_folder = os.path.join(output_folder, 'hallucinations')
correct_classifications_folder = os.path.join(output_folder, 'correct_classifications')
misclassifications_folder = os.path.join(output_folder, 'misclassifications')
wrong_objects_folder = os.path.join(output_folder, 'wrong_objects')

os.makedirs(hallucinations_folder, exist_ok=True)
os.makedirs(correct_classifications_folder, exist_ok=True)
os.makedirs(misclassifications_folder, exist_ok=True)
os.makedirs(wrong_objects_folder, exist_ok=True)

# Define a function to draw bounding boxes on an image
def draw_bboxes(image, bboxes, color):
    draw = ImageDraw.Draw(image)
    for bbox, label in bboxes:
        if label.lower() != 'unknown':
            left, top, width, height = bbox
            right = left + width
            bottom = top + height
            draw.rectangle([left, top, right, bottom], outline=color, width=3)
            draw.text((left, top), label, fill=color)

# Define a function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

# Function to parse bounding box from string
def parse_bounding_box(bbox_str):
    bbox_str = bbox_str.split('[')[-1].split(']')[0]
    bbox = list(map(float, bbox_str.split(',')))
    return bbox

# Lists to store classification results
hallucinations = []
correctly_classified = []
misclassified = []
wrong_objects = []

# Process each image in the images folder
for image_filename in os.listdir(images_folder):
    if image_filename.endswith('.jpg'):
        image_id = image_filename.split('.')[0]
        image_path = os.path.join(images_folder, image_filename)
        image = Image.open(image_path)
        orig_size = image.size

        # Get the bounding boxes
        predicted_boxes = [bbox for bbox in predicted_bboxes if bbox['image'] == image_filename]
        gt_boxes = [bbox for bbox in ground_truth_bboxes if bbox['image'] == image_filename]

        # Extract bounding boxes from the data
        predicted_bboxes_list = []
        for item in predicted_boxes:
            for obj in item['bbx with object']:
                obj_name, bbox = obj.split(' [')
                bbox = bbox.strip(']').split(', ')
                bbox = [float(coord) for coord in bbox]
                predicted_bboxes_list.append((bbox, obj_name))

        gt_bboxes_list = []
        for item in gt_boxes:
            for obj in item['bbx with object']:
                obj_name, bbox = obj.split(' [')
                bbox = bbox.strip(']').split(', ')
                bbox = [float(coord) for coord in bbox]
                gt_bboxes_list.append((bbox, obj_name))

        # Check if there are no predicted bounding boxes
        if not predicted_bboxes_list:
            # Draw the ground truth bounding boxes (green)
            draw_bboxes(image, gt_bboxes_list, 'green')
            iou_text = ["no objects found by CogVLM"]

            # Save the image with the message
            output_image_path = os.path.join(wrong_objects_folder, f'{image_id}.jpg')
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'{image_id}\n' + '\n'.join(iou_text))
            plt.savefig(output_image_path, bbox_inches='tight')
            plt.close()
            continue

        # Draw the ground truth bounding boxes (green)
        draw_bboxes(image, gt_bboxes_list, 'green')

        # Draw the predicted bounding boxes (red)
        draw_bboxes(image, predicted_bboxes_list, 'red')

        # Calculate IoU values
        iou_text = []
        for pred_box in predicted_bboxes_list:
            pred_box_parsed, pred_label = pred_box
            max_iou = 0
            matched_gt_idx = -1
            for idx, gt_box in enumerate(gt_bboxes_list):
                gt_box_parsed, gt_label = gt_box
                iou = calculate_iou(pred_box_parsed, gt_box_parsed)
                if iou > max_iou:
                    max_iou = iou
                    matched_gt_idx = idx
            
            matched_gt_label = gt_bboxes_list[matched_gt_idx][1] if matched_gt_idx >= 0 else None
            iou_text.append(f'{pred_label}: {max_iou:.2f}')
            
            if max_iou < 0.5:
                # If IoU is less than 0.5, classify as a hallucination
                if matched_gt_label == pred_label:
                    hallucinations.append({"image_id": image_id, "pred_box": pred_box, "iou": max_iou})
                    output_image_path = os.path.join(hallucinations_folder, f'{image_id}.jpg')
                else:
                    wrong_objects.append({"image_id": image_id, "pred_box": pred_box, "iou": max_iou})
                    output_image_path = os.path.join(wrong_objects_folder, f'{image_id}.jpg')
            else:
                # Check if the labels match
                if matched_gt_label == pred_label:
                    correctly_classified.append({"image_id": image_id, "pred_box": pred_box, "iou": max_iou, "label": pred_label})
                    output_image_path = os.path.join(correct_classifications_folder, f'{image_id}.jpg')
                else:
                    misclassified.append({"image_id": image_id, "pred_box": pred_box, "iou": max_iou, "pred_label": pred_label, "gt_label": matched_gt_label})
                    output_image_path = os.path.join(misclassifications_folder, f'{image_id}.jpg')

        # Save the image with IoU values
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'{image_id}\n' + '\n'.join(iou_text))
        plt.savefig(output_image_path, bbox_inches='tight')
        plt.close()