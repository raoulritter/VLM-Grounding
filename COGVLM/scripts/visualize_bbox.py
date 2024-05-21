import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# Load the JSON data
with open('VLM-Grounding/COGVLM/data/bbxes_objects/bboxes_objects.json', 'r') as file:
    predicted_bboxes = json.load(file)

with open('VLM-Grounding/COGVLM/data/gt_bboxes/gt_bboxes.json', 'r') as file:
    ground_truth_bboxes = json.load(file)

# Define the list of images to process
image_ids = ["000000248334", "000000457884", "000000082812", "000000358195", "000000213224"]

# Function to adjust bounding boxes according to actual image dimensions
def adjust_bboxes(bboxes, orig_size, target_size=(1000, 1000)):
    orig_width, orig_height = orig_size
    adjusted_bboxes = []
    for bbox, label in bboxes:
        x, y, w, h = bbox
        # Scale the bounding box coordinates
        x = x / target_size[0] * orig_width
        y = y / target_size[1] * orig_height
        w = w / target_size[0] * orig_width
        h = h / target_size[1] * orig_height
        adjusted_bboxes.append(([x, y, w, h], label))
    return adjusted_bboxes

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

# Initialize a list to store IoU values
iou_data = []

# Ensure the output directory exists
output_dir = 'VLM-Grounding/COGVLM/data/visualize_bbox'
os.makedirs(output_dir, exist_ok=True)

# Process each image
for image_id in image_ids:
    # Load the image
    image_path = f'VLM-Grounding/COGVLM/data/images/{image_id}.jpg'
    image = Image.open(image_path)
    orig_size = image.size

    # Get the bounding boxes
    predicted_boxes = [bbox for bbox in predicted_bboxes if bbox['image'] == f'{image_id}.jpg']
    gt_boxes = [bbox for bbox in ground_truth_bboxes if bbox['image'] == f'{image_id}.jpg']

    # Extract and adjust bounding boxes from the data
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

    # Adjust predicted bounding boxes to actual image size
    predicted_bboxes_list = adjust_bboxes(predicted_bboxes_list, orig_size)

    # Draw the ground truth bounding boxes (green)
    draw_bboxes(image, gt_bboxes_list, 'green')

    # Draw the predicted bounding boxes (red)
    draw_bboxes(image, predicted_bboxes_list, 'red')

    # Calculate IoU values
    for gt_bbox, gt_label in gt_bboxes_list:
        for pred_bbox, pred_label in predicted_bboxes_list:
            if gt_label == pred_label:
                iou = calculate_iou(gt_bbox, pred_bbox)
                iou_data.append({
                    'Image ID': image_id,
                    'Object': gt_label,
                    'IoU': iou
                })

    # Save the image with bounding boxes
    save_path = os.path.join(output_dir, f'{image_id}.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(image_id)
    plt.savefig(save_path)
    plt.close()

# # Convert IoU data to a DataFrame and display it
# iou_df = pd.DataFrame(iou_data)
# display(iou_df)
# print(iou_df)