import ast
import cv2
import numpy as np
import pandas as pd
import pytesseract
import re

# Function to draw borders on detected objects
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=5, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Function to format plate number to Indonesian format
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
            formatted_plate = f"{prefix} {numbers} {suffix}"
            print(f"Formatted plate number: {formatted_plate}")
            return formatted_plate
    
    parts = re.findall(r'[A-Z]+|[0-9]+', text)
    if len(parts) >= 3:
        prefix = validate_section(parts[0], 'alpha')
        numbers = validate_section(parts[1], 'num')
        suffix = validate_section(''.join(parts[2:]), 'alpha')
        
        if len(prefix) <= 2 and len(numbers) <= 4 and len(suffix) <= 3:
            formatted_plate = f"{prefix} {numbers} {suffix}"
            print(f"Formatted plate number: {formatted_plate}")
            return formatted_plate
    
    print("No valid Indonesian plate format found")
    return None

# Reading detection results
results = pd.read_csv('./detection_results_interpolated.csv')

# Open the video file
video_path = 'sample2.mp4'
cap = cv2.VideoCapture(video_path)

# Set up the video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./tracking_output3.mp4', fourcc, fps, (width, height))

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Process the video frame by frame
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    
    if ret:
        # Get detection data for the current frame
        df_ = results[results['frame_nmr'] == frame_nmr]
        
        for row_indx in range(len(df_)):
            # Get vehicle bounding box coordinates
            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = ast.literal_eval(
                df_.iloc[row_indx]['vehicle_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            )
            draw_border(
                frame, 
                (int(vehicle_x1), int(vehicle_y1)), 
                (int(vehicle_x2), int(vehicle_y2)), 
                (0, 255, 0), 
                25,
                line_length_x=200, 
                line_length_y=200
            )

            # Get license plate bounding box coordinates
            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            )
            cv2.rectangle(
                frame, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                (0, 0, 255), 
                12
            )

            # Extract license plate text using OCR
            plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
            plate_text = pytesseract.image_to_string(plate_img, config='--psm 8')
            print(f"Detected Plate Text: {plate_text}")

            # Format the Indonesian plate number
            formatted_plate = format_plate_number(plate_text.strip())
            if formatted_plate:
                # Replace this line:
                cv2.putText(
                    frame,
                    f"Plate: {formatted_plate}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # Smaller text size (reduce the font scale)
                    (0, 255, 0),  # Black color
                    2,  # Boldness (increase thickness)
                    cv2.LINE_AA  # Anti-aliased line for better text quality
                )

        out.write(frame)

# Save and close the video output
out.release()
cap.release()
cv2.destroyAllWindows()
