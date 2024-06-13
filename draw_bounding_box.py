import cv2
import numpy as np
import random

# # Function to read bounding box information from the txt file
# def read_bbox_info(file_path):
#     bbox_info = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             data = line.strip().split(',')
#             frame_id = int(data[0])
#             object_id = int(data[1])
#             x, y, w, h = map(int, data[2:6])
#             bbox_info.append((frame_id, object_id, x, y, w, h))
#     return bbox_info

# # Function to draw bounding boxes on images with different colors for each object
# def draw_bounding_boxes(image_folder, bbox_info):
#     images_with_boxes = []
#     colors = {}  # Dictionary to store colors for each object ID
#     for frame_id, object_id, x, y, w, h in bbox_info:
#         image_path = f"{image_folder}/{frame_id:08d}.jpg"  # Assuming image filenames are in the format frame_00001.jpg, frame_00002.jpg, etc.
#         image = cv2.imread(image_path)
        
#         # Generate a unique color for each object ID if not already assigned
#         if object_id not in colors:
#             colors[object_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
#         # Draw bounding box with the assigned color
#         cv2.rectangle(image, (x, y), (x + w, y + h), colors[object_id], 2)
#         images_with_boxes.append(image)
#     return images_with_boxes
def read_bbox_info(file_path):
    bbox_info = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            frame_id = int(data[0])
            object_id = int(data[1])
            x, y, w, h = map(float, data[2:6])
            x,y,w,h = int(x),int(y),int(w),int(h),
            if frame_id not in bbox_info:
                bbox_info[frame_id] = []
            bbox_info[frame_id].append((object_id, x, y, w, h))
    return bbox_info

# Function to draw bounding boxes for all object IDs in a frame with different colors
def draw_bounding_boxes(image_folder, bbox_info):
    images_with_boxes = []
    colors = {}
    for frame_id, objects_info in bbox_info.items():
        image_path = f"{image_folder}/{frame_id:08d}.jpg"  # Assuming image filenames are in the format frame_00001.jpg, frame_00002.jpg, etc.
        image = cv2.imread(image_path)
        for object_id, x, y, w, h in objects_info:
            # Generate a unique color for each object ID
            if object_id not in colors:
                colors[object_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # Draw bounding box with the assigned color
            cv2.rectangle(image, (x, y), (x + w, y + h), colors[object_id], 2)
        images_with_boxes.append(image)
    return images_with_boxes
# Function to create video from images with bounding boxes
def create_video(images_with_boxes, output_video_path, fps=25):
    height, width, _ = images_with_boxes[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image in images_with_boxes:
        video_writer.write(image)
    video_writer.release()

# Path to the folder containing images
image_folder = "/mnt/DATA/EE22M204/Downloads/OCL_DEEP_OC_SORT-exp/data/mot/train/MOT17-02-FRCNN/img1"
# Path to the bbox.txt file
bbox_file = "/mnt/DATA/EE22M204/Downloads/OCL_DEEP_OC_SORT-exp/results/trackers/MOT17-val/track_giou_follow_post/data/MOT17-02-FRCNN.txt"
# Path to save the output video
output_video_path = "/mnt/DATA/EE22M204/Downloads/OCL_DEEP_OC_SORT-exp/output_video_pred.avi"

# Read bounding box information
bbox_info = read_bbox_info(bbox_file)

# Draw bounding boxes on images with different colors for each object
images_with_boxes = draw_bounding_boxes(image_folder, bbox_info)

# Create video from images with bounding boxes
create_video(images_with_boxes, output_video_path)
