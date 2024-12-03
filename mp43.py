from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
import torch
from torchvision.ops import nms
import re
from difflib import get_close_matches
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def region(file_path):
    region_codes = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  
                code, region = line.strip().split(': ')
                region_codes[code] = region
    return region_codes

REGION_CODES = region('region.txt')

def enhance_plate_image(plate_img):
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def get_multiple_plate_readings(plate_img):
    readings = []
    
    # Original image
    result = ocr.ocr(plate_img, cls=True)
    if result[0]:
        readings.extend([(line[1][0], line[1][1]) for line in result[0]])
    
    # Enhanced image
    enhanced = enhance_plate_image(plate_img)
    result = ocr.ocr(enhanced, cls=True)
    if result[0]:
        readings.extend([(line[1][0], line[1][1]) for line in result[0]])
    
    # Rotated variations
    for angle in [-5, 5]:
        height, width = plate_img.shape[:2]
        matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated = cv2.warpAffine(plate_img, matrix, (width, height))
        result = ocr.ocr(rotated, cls=True)
        if result[0]:
            readings.extend([(line[1][0], line[1][1]) for line in result[0]])
    
    return readings

def validate_plate_format(text):
    pattern = r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$'
    return bool(re.match(pattern, text))

def get_closest_region_code(code):
    if code in REGION_CODES:
        return code
    matches = get_close_matches(code, REGION_CODES.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def format_plate_number(text):
    if not text:
        return None
    
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
                result += char_substitutions.get(char, char)
            elif expected_type == 'num' and not char.isdigit():
                result += char_substitutions.get(char, char)
            else:
                result += char
        return result

    # Try to match number pattern
    number_match = re.search(r'[0-9]{1,4}', text)
    if number_match:
        number_start = number_match.start()
        number_end = number_match.end()
        
        prefix = validate_section(text[:number_start], 'alpha')
        numbers = validate_section(text[number_start:number_end], 'num')
        suffix = validate_section(text[number_end:], 'alpha')
        
        if len(prefix) <= 2 and len(numbers) <= 4 and len(suffix) <= 3:
            region_check = get_closest_region_code(prefix)
            if region_check:
                return f"{region_check} {numbers} {suffix}"

    return None

def apply_nms(detections, conf_threshold=0.5, iou_threshold=0.4):
    if not detections:
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
    return [detections[i] for i in indices]

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
                if all(k in results[frame_nmr][vehicle_id] for k in ['vehicle', 'license_plate']):
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        vehicle_id,
                        '[{} {} {} {}]'.format(*results[frame_nmr][vehicle_id]['vehicle']['bbox']),
                        '[{} {} {} {}]'.format(*results[frame_nmr][vehicle_id]['license_plate']['bbox']),
                        results[frame_nmr][vehicle_id]['license_plate']['bbox_score'],
                        results[frame_nmr][vehicle_id]['license_plate']['text'],
                        results[frame_nmr][vehicle_id]['license_plate']['text_score']
                    ))

# Main execution
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
                plate_readings = get_multiple_plate_readings(license_plate_crop)
                
                if plate_readings:
                    best_reading = max(plate_readings, key=lambda x: x[1])
                    license_plate_text = best_reading[0]
                    license_plate_text_score = best_reading[1]
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
                elif formatted_plate_text:
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