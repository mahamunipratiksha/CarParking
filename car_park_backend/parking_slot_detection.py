from ultralytics import YOLO
import numpy as np
import os
import cv2
import cv2
from datetime import datetime
from tqdm import tqdm
import shutil
import json

model = YOLO("models/yolov8n_fold_1.pt",verbose=False)

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'X264')  # or 'H264'
    writer = cv2.VideoWriter(output_filename, fourcc, float(fps), (frame_width, frame_height))


    return writer


def save_yolo_pred(predictions,output_file_path):
    with open(output_file_path, 'w') as file:
        for idx, prediction in enumerate(predictions[0].boxes.xywhn):
            cls = int(predictions[0].boxes.cls[idx].item())
            file.write(f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, annotation_path, threshold, highlighted_cars):

    # Read the annotation file
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    # Process each annotation and store rectangle information in a list
    rectangles = []
    for annotation in annotations:
        # Parse the annotation values
        class_id, x_center, y_center, width, height = map(float, annotation.split())

        if class_id != 0:

            # Calculate the bounding box coordinates
            left = int((x_center - width / 2) * image.shape[1])
            top = int((y_center - height / 2) * image.shape[0])
            right = int((x_center + width / 2) * image.shape[1])
            bottom = int((y_center + height / 2) * image.shape[0])

            # Check if there is a car occupying the parking spot
            if is_occupied(image, annotations, left, top, right, bottom, threshold):
                color = (0, 0, 255)  # Red color
            else:
                color = (0, 255, 0)  # Green color


            # Store the rectangle information in the list
            rectangles.append((left, top, right, bottom, color, class_id))

    d = {}
    # Draw the bounding box rectangles on the image
    for rectangle in rectangles:
        left, top, right, bottom, color, class_id = rectangle
        cv2.rectangle(image, (left, top), (right, bottom), color, 2) # type: ignore
        cv2.putText(image, str(int(class_id)), ((left + right) // 2, (top + bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        # cv2.putText(image, str(class_id), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if color[1] == 255:
            cls_label = 1
        else:
            cls_label = 0

        d[class_id] = cls_label

    return d,image

# Function to check if a car is occupying a parking spot
def is_occupied(image, annotations, left, top, right, bottom, threshold):
    # Get the car bounding boxes
    car_boxes = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.split())
        if class_id == 0:
            car_left = int((x_center - width / 2) * image.shape[1])
            car_top = int((y_center - height / 2) * image.shape[0])
            car_right = int((x_center + width / 2) * image.shape[1])
            car_bottom = int((y_center + height / 2) * image.shape[0])
            car_boxes.append((car_left, car_top, car_right, car_bottom))

    # If there are no cars in the image that intersect with the parking spot, return False
    if not car_boxes:
        return False

    # Calculate the intersection over union (IoU) between the parking spot and each car
    ious = []
    for car_box in car_boxes:
        intersection_left = max(left, car_box[0])
        intersection_top = max(top, car_box[1])
        intersection_right = min(right, car_box[2])
        intersection_bottom = min(bottom, car_box[3])
        intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)
        parking_spot_area = (right - left) * (bottom - top)
        car_area = (car_box[2] - car_box[0]) * (car_box[3] - car_box[1])
        iou = intersection_area / (parking_spot_area + car_area - intersection_area)
        ious.append(iou)

    # Return True if the maximum IoU is above the threshold, False otherwise
    return np.max(ious) > threshold

def frame_prediction(vid_frame,parking_slot_coordinates_file_path,threshold=0.3, highlighted_cars=True):
    results = model(vid_frame, save_txt=None,verbose=False)
    save_yolo_pred(results,'res.txt')

    annotation_path = 'merged_file.txt'
    merge_files(parking_slot_coordinates_file_path, 'res.txt', annotation_path)

    d = draw_bounding_boxes(vid_frame, annotation_path, threshold, highlighted_cars)
    return d

def merge_files(file1_path, file2_path, output_path):
    try:
        # Read content from the first file
        with open(file1_path, 'r') as file1:
            content1 = file1.read()

        # Read content from the second file
        with open(file2_path, 'r') as file2:
            content2 = file2.read()

        # Combine the content of both files
        merged_content = content1 + content2

        # Write the merged content to the output file
        with open(output_path, 'w') as output_file:
            output_file.write(merged_content)

        # print(f"Merged content successfully written to {output_path}")

    except FileNotFoundError:
        print("One or both input files not found.")


def get_vid_predictions(vid_path, labels_path, predict_interval):
    folder_path = os.path.join('results', datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    os.makedirs(folder_path)

    shutil.copy(vid_path, folder_path)
    shutil.copy(labels_path, folder_path)
    output_filename = os.path.join(folder_path, "output.mp4")
    video_path = vid_path
    parking_slot_coordinates_file_path = labels_path
    output_label_file_path = 'prediction.json'

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = create_video_writer(cap, output_filename)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    d = {}
    frame_count = 0
    i = 1

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while True:
            success, vid_frame = cap.read()

            if not success:
                break

            frame_count += 1

            r, result_img = frame_prediction(vid_frame, parking_slot_coordinates_file_path, threshold=0.3, highlighted_cars=True)
            writer.write(result_img)

            if frame_count % (fps * predict_interval) == 0:
                frame_count = 0
                d[i * predict_interval] = r
                i += 1

            pbar.update(1)

    cap.release()  # Release the video capture
    writer.release()  # Release the video writer

    with open(os.path.join(folder_path, output_label_file_path), 'w') as fp:
        json.dump(d, fp)

    return os.path.abspath(output_filename), os.path.abspath(os.path.join(folder_path, output_label_file_path))


if __name__ == '__main__':
    vid_path = 'lot.mp4'
    labels_path = 'label_file_path.txt'
    predict_interval = 1
    output_vid_path,output_pred_path = get_vid_predictions(vid_path,labels_path,predict_interval)
    print(output_vid_path,output_pred_path)

