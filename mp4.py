from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *

import torch
from torchvision.ops import nms
import string
import easyocr
import re
from difflib import get_close_matches

reader = easyocr.Reader(['en'], gpu=False)

def region(file_path):
    region_codes = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  
                code, region = line.strip().split(': ')
                region_codes[code] = region
    return region_codes

REGION_CODES = region('region.txt')

print(REGION_CODES)

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

    for vehicle in vehicle_track_ids:
        xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_id = vehicle

        if x1 > xvehicle1 and y1 > yvehicle1 and x2 < xvehicle2 and y2 < yvehicle2:
            return vehicle

    return -1, -1, -1, -1, -1

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'vehicle_id', 'vehicle_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for vehicle_id in results[frame_nmr].keys():
                if 'vehicle' in results[frame_nmr][vehicle_id].keys() and \
                   'license_plate' in results[frame_nmr][vehicle_id].keys() and \
                   'text' in results[frame_nmr][vehicle_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            vehicle_id,
                                                            '[{} {} {} {}]'.format(
                                                                *results[frame_nmr][vehicle_id]['vehicle']['bbox']),
                                                            '[{} {} {} {}]'.format(
                                                                *results[frame_nmr][vehicle_id]['license_plate']['bbox']),
                                                            results[frame_nmr][vehicle_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][vehicle_id]['license_plate']['text'],
                                                            results[frame_nmr][vehicle_id]['license_plate']['text_score']))

def enhance_plate_image(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    denoised = cv2.fastNlMeansDenoising(gray)
    
    binary = cv2.adaptiveThreshold(denoised, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def format_plate_number(text):
    print(f"Formatting plate number: {text}")
    
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    char_substitutions = {
        'O': '0', '0': 'O',
        'I': '1', '1': 'I',
        'Z': '2', '2': 'Z',
        'J': '3', '3': 'J',
        'A': '4', '4': 'A',
        'S': '5', '5': 'S',
        'G': '6', '6': 'G',
        'T': '7', '7': 'T',
        'B': '8', '8': 'B',
    }
    
    def validate_section(text_part, expected_type='alpha'):
        result = ''
        for char in text_part:
            if expected_type == 'alpha' and char.isdigit():
                if char in char_substitutions:
                    result += char_substitutions[char]
                else:
                    result += char
            elif expected_type == 'num' and not char.isdigit():
                if char in char_substitutions:
                    result += char_substitutions[char]
                else:
                    result += char
            else:
                result += char
        return result
    
    number_match = re.search(r'[0-9]{1,4}', text)
    if number_match:
        number_start = number_match.start()
        number_end = number_match.end()
        
        prefix = text[:number_start]
        numbers = text[number_start:number_end]
        suffix = text[number_end:]
        
        prefix = validate_section(prefix, 'alpha')
        numbers = validate_section(numbers, 'num')
        suffix = validate_section(suffix, 'alpha')
        
        if len(prefix) <= 2 and len(numbers) <= 4 and len(suffix) <= 3:
            region_check = get_closest_region_code(prefix)
            if region_check:
                formatted_plate = f"{region_check} {numbers} {suffix}"
                print(f"Formatted plate number: {formatted_plate}")
                return formatted_plate
    
    parts = re.findall(r'[A-Z]+|[0-9]+', text)
    if len(parts) >= 3:
        prefix = validate_section(parts[0], 'alpha')
        numbers = validate_section(parts[1], 'num')
        suffix = validate_section(''.join(parts[2:]), 'alpha')
        
        if len(prefix) <= 2 and len(numbers) <= 4 and len(suffix) <= 3:
            region_check = get_closest_region_code(prefix)
            if region_check:
                formatted_plate = f"{region_check} {numbers} {suffix}"
                print(f"Formatted plate number: {formatted_plate}")
                return formatted_plate
    
    print("No valid Indonesian plate format found")
    return None

def get_closest_region_code(code):
    print(f"Getting closest region code for '{code}'")
    if code in REGION_CODES:
        print(f"Exact match found: {code}")
        return code
    matches = get_close_matches(code, REGION_CODES.keys(), n=1, cutoff=0.6)
    closest_match = matches[0] if matches else None
    if closest_match:
        print(f"Closest match found: {closest_match}")
    else:
        print("No close match found.")
    return closest_match


# masi belum kepake, coba diintegrasiin siapa tau hasilnya lebih bagus
def read_plate(reader, plate_img):
    print("Reading plate text using OCR.")
    
    rows, cols = plate_img.shape[:2]
    if rows > cols:
        plate_img = cv2.rotate(plate_img, cv2.ROTATE_90_CLOCKWISE)
    
    attempts = [
        plate_img,  
        enhance_plate_image(plate_img),  
        cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY),  
        cv2.threshold(cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY), 
                     0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  
    ]
    
    all_texts = []
    
    for img in attempts:
        results = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        if results:
            for (bbox, text, conf) in results:
                all_texts.append((text, conf))
            
            if len(results) > 1:
                results.sort(key=lambda x: x[0][0][0])  
                combined_text = ''.join(r[1] for r in results)
                avg_conf = sum(r[2] for r in results) / len(results)
                all_texts.append((combined_text, avg_conf))
    
    all_texts.sort(key=lambda x: x[1], reverse=True)
    
    for text, conf in all_texts:
        formatted = format_plate_number(text)
        if formatted:
            print(f"Successfully detected plate: {formatted}")
            return formatted
    
    print("No valid plate text detected")
    return "Tidak Terbaca"

results = {}
valid_license_plates = {}  
mot_tracker = Sort()

coco_model = YOLO('./model/yolo11n.pt')
license_plate_detector = YOLO('./model/best.pt')

cap = cv2.VideoCapture('sample2.mp4')

vehicles = [2, 3, 5, 7]

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        detections = coco_model(frame)[0]
        detections_ = [d[:5] for d in detections.boxes.data.tolist() if int(d[5]) in vehicles]

        track_ids = mot_tracker.update(np.array(detections_))

        license_plates = license_plate_detector(frame)[0]
        plates_detections = apply_nms([lp[:5] for lp in license_plates.boxes.data.tolist()])

        for license_plate in plates_detections:
            x1, y1, x2, y2, score = license_plate
            xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_id = get_vehicle(license_plate, track_ids)

            if vehicle_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                license_plate_texts = reader.readtext(license_plate_crop)

                if license_plate_texts:
                    best_text_data = max(license_plate_texts, key=lambda x: x[2])
                    license_plate_text = best_text_data[1]
                    license_plate_text_score = best_text_data[2]

                    formatted_plate_text = format_plate_number(license_plate_text)
                else:
                    license_plate_text = "Tidak Terbaca"
                    license_plate_text_score = 0
                    formatted_plate_text = None

                if vehicle_id in valid_license_plates:
                    previous_plate, previous_score = valid_license_plates[vehicle_id]
                    if formatted_plate_text and license_plate_text_score > previous_score:
                        valid_license_plates[vehicle_id] = (formatted_plate_text, license_plate_text_score)
                    else:
                        formatted_plate_text = previous_plate  
                else:
                    if formatted_plate_text:
                        valid_license_plates[vehicle_id] = (formatted_plate_text, license_plate_text_score)

                results[frame_nmr][vehicle_id] = {
                    'vehicle': {'bbox': [xvehicle1, yvehicle1, xvehicle2, yvehicle2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'bbox_score': score,
                        'text': formatted_plate_text if formatted_plate_text else license_plate_text,
                        'text_score': license_plate_text_score
                    }
                }



write_csv(results, './hasil_plat_indo.csv')
