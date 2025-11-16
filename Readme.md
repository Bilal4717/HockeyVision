🎯 HockeyVision — Player & Puck Tracking with YOLO + ByteTrack

HockeyVision is a computer-vision prototype that detects players, tracks the puck, and generates advanced match analytics including:

Player speed

Distance covered

Puck possession

Touch counts

Shot detection

Shot-on-target accuracy

Annotated match video

CSV + Pickle reports

Built using YOLOv8, ByteTrack, OpenCV, and Supervision.

🚀 Features
✔ Player Tracking

Using YOLOv8n (COCO) → detects person
ByteTrack → assigns persistent IDs

✔ Puck Detection

Using yolov8m_forzasys_hockey_Version_2.pt (class ID = 0)

✔ Integrated Player + Puck Analytics

Possession (seconds)

Touches

Shots

Shots on target

Shot accuracy (%)

Distance & speed estimation

✔ Rink Region Filtering

Removes false detections outside the play area.

✔ Smoothed Puck Tracking

Prevents flickering & false positives.