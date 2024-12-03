import ast
import cv2
import numpy as np
import pandas as pd

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

results = pd.read_csv('./hasil_plat_indo_interpolated3.csv')

video_path = 'sample2.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./output-paddle.mp4', fourcc, fps, (width, height))

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        
        for row_indx in range(len(df_)):
            # Vehicle bbox
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

            # License plate bbox
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
            
            # Display the license plate text or text score
            license_number = df_.iloc[row_indx]['license_number']
            text_score = df_.iloc[row_indx]['license_number_score']
            
            # Display the license plate text (if exists)
            if license_number:
                cv2.putText(
                    frame,
                    f"LP: {license_number}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            # Display the text score (if available)
            if text_score:
                cv2.putText(
                    frame,
                    f"Score: {text_score}",
                    (int(x1), int(y2) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        # Write the frame to the output video
        out.write(frame)

out.release()
cap.release()
cv2.destroyAllWindows()
