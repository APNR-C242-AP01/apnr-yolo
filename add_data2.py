import csv
import numpy as np
from scipy.interpolate import interp1d
import re

def clean_value(value):
    # Menghapus prefix np.float64( dan suffix )
    cleaned_value = re.sub(r'np\.float64\((.*?)\)', r'\1', value)
    # Menghapus tanda koma yang tidak perlu
    cleaned_value = cleaned_value.replace(',', '').strip()  # Strip any surrounding spaces and remove commas
    return float(cleaned_value)

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    vehicle_ids = np.array([int(float(row['vehicle_id'])) for row in data])
    
    # Membersihkan nilai vehicle_bbox
    vehicle_bboxes = np.array([list(map(float, [clean_value(val) for val in row['vehicle_bbox'][1:-1].split()])) for row in data])
    
    # Mengambil license_plate_bbox dan teksnya
    license_plate_bboxes = np.array([list(map(float, [clean_value(val) for val in row['license_plate_bbox'][1:-1].split()])) for row in data])
    license_plate_texts = [row['license_number'] for row in data]
    
    # Menangani license_number_score yang kosong atau tidak valid
    license_number_scores = []
    for row in data:
        try:
            license_number_score = float(row['license_number_score']) if row['license_number_score'] else 0.0  # Default to 0.0 if empty
        except ValueError:
            license_number_score = 0.0  # Default to 0.0 if there's an error converting
        license_number_scores.append(license_number_score)
    
    interpolated_data = []
    unique_vehicle_ids = np.unique(vehicle_ids)
    
    for vehicle_id in unique_vehicle_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['vehicle_id'])) == int(float(vehicle_id))]
        print(f"Processing vehicle ID: {vehicle_id}, Frames: {frame_numbers_}")

        vehicle_mask = vehicle_ids == vehicle_id
        vehicle_frame_numbers = frame_numbers[vehicle_mask]
        vehicle_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []
        license_number_interpolated = []
        license_number_score_interpolated = []  # List for interpolating license_number_score

        first_frame_number = vehicle_frame_numbers[0]
        
        for i in range(len(vehicle_bboxes[vehicle_mask])):
            frame_number = vehicle_frame_numbers[i]
            vehicle_bbox = vehicle_bboxes[vehicle_mask][i]
            license_plate_bbox = license_plate_bboxes[vehicle_mask][i]
            license_number = license_plate_texts[vehicle_mask.nonzero()[0][i]]  # Gunakan indeks integer dari vehicle_mask
            license_number_score = license_number_scores[vehicle_mask.nonzero()[0][i]]  # Get the license_number_score for the current vehicle

            if i > 0:
                prev_frame_number = vehicle_frame_numbers[i-1]
                prev_vehicle_bbox = vehicle_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]
                prev_license_number = license_number_interpolated[-1]
                prev_license_number_score = license_number_score_interpolated[-1]  # Previous license_number_score

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    
                    interp_func = interp1d(x, np.vstack((prev_vehicle_bbox, vehicle_bbox)), axis=0, kind='linear')
                    interpolated_vehicle_bboxes = interp_func(x_new)
                    
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    vehicle_bboxes_interpolated.extend(interpolated_vehicle_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])
                    
                    license_number_interpolated.extend([license_number] * (frames_gap - 1))
                    license_number_score_interpolated.extend([license_number_score] * (frames_gap - 1))  # Interpolating license_number_score

            vehicle_bboxes_interpolated.append(vehicle_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)
            license_number_interpolated.append(license_number)
            license_number_score_interpolated.append(license_number_score)  # Adding current license_number_score

        for i in range(len(vehicle_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {
                'frame_nmr': str(frame_number),
                'vehicle_id': str(vehicle_id),
                'vehicle_bbox': ' '.join(map(str, vehicle_bboxes_interpolated[i])),
                'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i])),
                'license_number': license_number_interpolated[i],
                'license_plate_bbox_score': '0' if str(frame_number) not in frame_numbers_ else \
                    [p['license_plate_bbox_score'] for p in data if int(p['frame_nmr']) == frame_number and int(float(p['vehicle_id'])) == int(float(vehicle_id))][0],
                'license_number_score': str(license_number_score_interpolated[i])  # Adding license_number_score to the row
            }
            interpolated_data.append(row)

    return interpolated_data

with open('modify.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

interpolated_data = interpolate_bounding_boxes(data)

header = ['frame_nmr', 'vehicle_id', 'vehicle_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('hasil_plat_indo_interpolated3.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
