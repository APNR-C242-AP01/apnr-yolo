from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *

import torch
from torchvision.ops import nms

def apply_nms(detections, conf_threshold=0.5, iou_threshold=0.4):
    if len(detections) == 0:
        return []

    boxes = torch.tensor([d[0:4] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in detections], dtype=torch.float32)

    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]

    if boxes.ndimension() == 1:
        boxes = boxes.unsqueeze(0)

    if boxes.size(0) == 0:
        return []

    indices = nms(boxes, scores, iou_threshold)

    filtered_detections = [detections[i] for i in indices]
    return filtered_detections



def get_vehicle(license_plate, vehicle_track_ids, score_threshold=0.5):
    x1, y1, x2, y2, score = license_plate  

    if score < score_threshold:  
        return -1, -1, -1, -1, -1

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_id = vehicle_track_ids[j]

        if x1 > xvehicle1 and y1 > yvehicle1 and x2 < xvehicle2 and y2 < yvehicle2:
            vehicle_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[vehicle_indx]

    return -1, -1, -1, -1, -1


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{}\n'.format('frame_nmr', 'vehicle_id', 'vehicle_bbox',
                                         'license_plate_bbox', 'license_plate_bbox_score'))

        for frame_nmr in results.keys():
            for vehicle_id in results[frame_nmr].keys():
                if 'vehicle' in results[frame_nmr][vehicle_id].keys() and \
                   'license_plate' in results[frame_nmr][vehicle_id].keys():
                    f.write('{},{},{},{},{}\n'.format(frame_nmr,
                                                     vehicle_id,
                                                     '[{} {} {} {}]'.format(
                                                         results[frame_nmr][vehicle_id]['vehicle']['bbox'][0],
                                                         results[frame_nmr][vehicle_id]['vehicle']['bbox'][1],
                                                         results[frame_nmr][vehicle_id]['vehicle']['bbox'][2],
                                                         results[frame_nmr][vehicle_id]['vehicle']['bbox'][3]),
                                                     '[{} {} {} {}]'.format(
                                                         results[frame_nmr][vehicle_id]['license_plate']['bbox'][0],
                                                         results[frame_nmr][vehicle_id]['license_plate']['bbox'][1],
                                                         results[frame_nmr][vehicle_id]['license_plate']['bbox'][2],
                                                         results[frame_nmr][vehicle_id]['license_plate']['bbox'][3]),
                                                     results[frame_nmr][vehicle_id]['license_plate']['bbox_score'])
                          )
        f.close()

results = {}
mot_tracker = Sort()

coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('best.pt')

cap = cv2.VideoCapture('./sample2.mp4')

vehicles = [2, 3, 5, 7]  

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        track_ids = mot_tracker.update(np.asarray(detections_))
        
        license_plates = license_plate_detector(frame)[0]
        plates_detections = []
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            plates_detections.append([x1, y1, x2, y2, score])

        plates_detections = apply_nms(plates_detections)

        for license_plate in plates_detections:
            x1, y1, x2, y2, score = license_plate
            xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_id = get_vehicle(license_plate, track_ids)
            
            if vehicle_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                results[frame_nmr][vehicle_id] = {
                    'vehicle': {'bbox': [xvehicle1, yvehicle1, xvehicle2, yvehicle2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'bbox_score': score
                    }
                }

write_csv(results, './detection_results.csv')
