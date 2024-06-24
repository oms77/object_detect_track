from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import math

model = YOLO(r"C:\Users\Om sinha\Downloads\invec.pt")
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(r"C:\Users\Om sinha\Downloads\istockphoto-866517852-640_adpp_is.mp4")
out = cv2.VideoWriter('v12.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15,(640,360))
ret=1
fps = cap.get(cv2.CAP_PROP_FPS)
time_interval = 1 / fps

# Real-world distance per pixel (example value, needs calibration)
real_world_distance_per_pixel = 0.05

while ret:
    ret, frame1 = cap.read()
    result = model(frame1)[0]
    detections = sv.Detections.from_ultralytics(result)

    labels = [
        f"#{result.names[class_id]}"
        for class_id in detections.class_id
    ]

    annotated_frame = bounding_box_annotator.annotate(
        frame1.copy(), detections=detections)
    fr1 = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)

    l = detections.xyxy
    print(l)
    for x in range(len(l)):
        b1 = [int(l[x][0]), int(l[x][1]), int(l[x][2]), int(l[x][3])]

    while cap.isOpened():
        ret,frame2 = cap.read()
        result = model(frame2)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = [
            f"#{result.names[class_id]}"
            for class_id in detections.class_id
        ]

        annotated_frame = bounding_box_annotator.annotate(
            frame2.copy(), detections=detections)
        fr2 = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)

        l1 = detections.xyxy
        for x in range(len(l1)):
            b2 = [int(l1[x][0]), int(l1[x][1]), int(l1[x][2]), int(l1[x][3])]


        def get_centre(bbox):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return cx, cy

        def calculate_distance(bbox1, bbox2):
            """Calculate the Euclidean distance between the centers of two bounding boxes."""
            cx1, cy1 = get_centre(bbox1)
            cx2, cy2 = get_centre(bbox2)
            distance = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
            return distance, (cx1, cy1), (cx2, cy2)


        distance, center1, center2 = calculate_distance(b1, b2)
        real_world_distance_per_pixel = 0.0005
        distance_meters = distance * real_world_distance_per_pixel

        speed = distance_meters // time_interval
        text = f"Speed: {speed}m/s"
        mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
        cv2.putText(fr1, text,mid_point,cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out.write(fr1)
        fr1 = fr2

cap.release()
out.release()
cv2.destroyAllWindows()