import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pickle
from math import hypot
import pandas as pd
from collections import defaultdict


player_path = "yolov8n.pt"   
puck_path = r"C:\Users\DELL\OneDrive\Desktop\HockeyVision\yolov8m_forzasys_hockey_Version_2.pt"  # for puck class
input_video = r"C:\Users\DELL\OneDrive\Desktop\HockeyVision\video_input.mp4"   
output_video = r"C:\Users\DELL\OneDrive\Desktop\HockeyVision\outputs\annotated_output2.mp4"
output_pickle = r"C:\Users\DELL\OneDrive\Desktop\HockeyVision\outputs\track_stubs.pkl"
output_csv = r"C:\Users\DELL\OneDrive\Desktop\HockeyVision\outputs\player_stats.csv"



RINK_COORDS = np.array([[-450, 710], [2030, 710], [948, 61], [352, 61]])

rink_width = 15.0
rink_height = 30.0

FRAME_SKIP = 2  
DOWNSCALE = None  

#Filters 
conf_player = 0.40
conf_puck = 0.20
min_puck_area = 20
max_puck_area = 4000
puck_control_distance = 60
puck_frames = 2
shot_speed_inc = 150.0
min_shot_speed_puck = 200.0

# Colors 
text = (0, 255, 255)   
id_box = (0, 0, 0)
id_text = (255, 255, 255)
puck_color = (255, 255, 0)   
ellipse_clor = (255, 0, 0)
# ---------------------------------------------------------

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def euclidean(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])

class PlayerPuckAnalyzer:
    def __init__(self, player_path, puck_path, rink_coords=None):
        print("Loading models...")
        self.player_model = YOLO(player_path)   
        self.puck_model = YOLO(puck_path)       
        self.player_tracker = sv.ByteTrack()
        self.puck_tracker = sv.ByteTrack()
        
        self.last_tracked_players = []
        self.last_tracked_pucks = []
        
        self.player_class_idx = self.find_class(self.player_model, candidates=("person","player"))
        self.puck_class_idx = self.find_class(self.puck_model, candidates=("puck","ball"))
        print(f" Player class idx: {self.player_class_idx}, Puck class idx: {self.puck_class_idx}")
        
        self.rink_coords = rink_coords
        self.rink_mask = None
        self.px_per_m = None
        if rink_coords is not None:
            self.compute_px()
        
        self.prev_positions = {}             
        self.cum_distance_px = defaultdict(float)
        self.player_stats = defaultdict(lambda: {"distance_m":0.0, "distance_px":0.0,
                                                 "possession_frames":0, "touches":0,
                                                 "shots":0, "shots_on_target":0})
       
        self.puck_prev_pos = None
        self.puck_prev_speed = 0.0
        self.puck_detection_count = 0
        self.puck_last_valid_frame = -9999
        self.prev_controller = None
       
        self.frames_records = []
        self.shot_events = []

    def find_class(self, model, candidates=("puck",)):
        names = model.names if hasattr(model, "names") else {}
        for idx, nm in names.items():
            nm_l = nm.lower()
            for cand in candidates:
                if cand in nm_l:
                    return idx
       
        return 0

    def compute_px(self):
       
        left_pixel = self.rink_coords[0][0]
        right_pixel = self.rink_coords[1][0]
        top_pixel = self.rink_coords[2][1]
        bottom_pixel = self.rink_coords[0][1]
        px_per_m_x = (right_pixel - left_pixel) / rink_width
        px_per_m_y = (bottom_pixel - top_pixel) / rink_height
        self.px_per_m = (abs(px_per_m_x) + abs(px_per_m_y))/2.0
        print(f"px_per_m ≈ {self.px_per_m:.2f}")

    def make_rink(self, frame_shape):
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.rink_coords.astype(int), 1)
        self.rink_mask = mask.astype(bool)

    def inside_rink(self, x, y):
        if self.rink_mask is None:
            return True
        h,w = self.rink_mask.shape
        if x<0 or x>=w or y<0 or y>=h:
            return False
        return bool(self.rink_mask[y,x])

    def process(self, input_video, output_path, frame_skip=FRAME_SKIP, downscale=DOWNSCALE):
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video: "+input_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.rink_coords is not None:
            self.make_rink((height, width, 3))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        print("Starting processing")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            
            frame_for_infer = frame
            scale_x = scale_y = 1.0
            if downscale is not None:
                
                small = cv2.resize(frame, downscale)
                frame_for_infer = small
                scale_x = frame.shape[1] / downscale[0]
                scale_y = frame.shape[0] / downscale[1]

            do_infer = (frame_idx % frame_skip == 0)


            if do_infer:
                res_p = self.player_model(frame_for_infer, conf=conf_player)[0]
                dets_p = sv.Detections.from_ultralytics(res_p)

                if downscale is not None and len(dets_p) > 0:
                    boxes = []
                    for xyxy in dets_p.xyxy:
                        x1,y1,x2,y2 = xyxy.cpu().numpy()
                        boxes.append([x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y])
                   
                    confs = dets_p.confidence if hasattr(dets_p, "confidence") else None
                    clsids = dets_p.class_id if hasattr(dets_p, "class_id") else None
                    dets_p = sv.Detections(xyxy=np.array(boxes),
                                           confidence=(confs.cpu().numpy() if hasattr(confs, "cpu") else confs),
                                           class_id=(clsids.cpu().numpy() if hasattr(clsids, "cpu") else clsids))
               
                try:
                    class_ids = dets_p.class_id
                except Exception:
                    class_ids = getattr(dets_p, "class_id", [])
                person_mask = [int(cid)==int(self.player_class_idx) for cid in class_ids] if len(class_ids)>0 else []
                if len(person_mask) > 0:
                    try:
                        dets_p = dets_p[person_mask]
                    except Exception:
                        pass
                tracked_players = self.player_tracker.update_with_detections(dets_p)
               
                self.last_tracked_players = tracked_players
            else:
                tracked_players = self.last_tracked_players

            #Puck Detection 
            if do_infer:
                res_b = self.puck_model(frame_for_infer, conf=conf_puck)[0]
                dets_b = sv.Detections.from_ultralytics(res_b)
               
                if downscale is not None and len(dets_b) > 0:
                    boxes = []
                    for xyxy in dets_b.xyxy:
                        x1,y1,x2,y2 = xyxy.cpu().numpy()
                        boxes.append([x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y])
                    confs = dets_b.confidence if hasattr(dets_b, "confidence") else None
                    clsids = dets_b.class_id if hasattr(dets_b, "class_id") else None
                    dets_b = sv.Detections(xyxy=np.array(boxes),
                                           confidence=(confs.cpu().numpy() if hasattr(confs, "cpu") else confs),
                                           class_id=(clsids.cpu().numpy() if hasattr(clsids, "cpu") else clsids))
                
                try:
                    class_ids_b = dets_b.class_id
                except Exception:
                    class_ids_b = getattr(dets_b, "class_id", [])
                puck_mask = [int(cid)==int(self.puck_class_idx) for cid in class_ids_b] if len(class_ids_b)>0 else []
                if len(puck_mask) > 0:
                    try:
                        dets_b = dets_b[puck_mask]
                    except Exception:
                        pass
                tracked_pucks = self.puck_tracker.update_with_detections(dets_b)
                self.last_tracked_pucks = tracked_pucks
            else:
                tracked_pucks = self.last_tracked_pucks

            # frame_players dict 
            frame_players = {}
            for tr in tracked_players:
                try:
                    bbox = tr[0].tolist()
                    track_id = int(tr[4])
                except Exception:
                    bbox = np.array(tr[0]).tolist()
                    track_id = int(getattr(tr, "track_id", -1))
                x1,y1,x2,y2 = map(int, bbox)
                cx,cy = get_center(bbox)
                 
                if not self.inside_rink(cx, cy):
                    continue
                prev = self.prev_positions.get(track_id, None)
                if prev is not None:
                    dist_px = euclidean((cx,cy), prev)
                else:
                    dist_px = 0.0
                self.prev_positions[track_id] = (cx,cy)
                self.cum_distance_px[track_id] = self.cum_distance_px.get(track_id, 0.0) + dist_px
                if self.px_per_m:
                    cum_m = self.cum_distance_px[track_id] / self.px_per_m
                    speed = (dist_px / self.px_per_m) * fps
                    speed_label = f"{speed:.2f} m/s"
                    dist_label = f"{cum_m:.2f} m"
                    self.player_stats[track_id]["distance_m"] = cum_m
                else:
                    speed_px_s = dist_px * fps
                    speed_label = f"{speed_px_s:.1f} px/s"
                    dist_label = f"{self.cum_distance_px[track_id]:.1f} px"
                    self.player_stats[track_id]["distance_px"] = self.cum_distance_px[track_id]
                frame_players[track_id] = {"bbox": bbox, "center": (cx,cy), "speed_label": speed_label, "dist_label": dist_label}

            #Puck filtering 
            valid_puck_found = False
            puck_center = None
            for tr in tracked_pucks:
                try:
                    bbox = tr[0].tolist()
                    puck_tid = int(tr[4])
                    score = float(tr[1]) if len(tr) > 1 else None
                except Exception:
                    bbox = np.array(tr[0]).tolist()
                    puck_tid = int(getattr(tr, "track_id", -1))
                    score = None
                x1,y1,x2,y2 = map(int, bbox)
                cx,cy = get_center(bbox)
                area = (x2-x1)*(y2-y1)
                # filter by rink membership
                if not self.inside_rink(cx, cy):
                    continue
                # filter by area
                if area < min_puck_area or area > max_puck_area:
                    continue
                # filter by score if available
                if (score is not None) and (score < conf_puck):
                    continue
                # accepted
                puck_center = (cx,cy)
                valid_puck_found = True
                cv2.circle(frame, puck_center, 8, puck_color, -1)
                break

            # temporal smoothing
            if valid_puck_found:
                self.puck_detection_count = getattr(self, "puck_detection_count", 0) + 1
                self.puck_last_valid_frame = frame_idx
                self.puck_prev_pos = puck_center
            else:
                # allow short occlusion 
                if (frame_idx - getattr(self, "puck_last_valid_frame", -9999)) <= 3:
                    puck_center = getattr(self, "puck_prev_pos", None)
                else:
                    puck_center = None
                    self.puck_detection_count = 0

            # possession / touches / shot detection 
            current_controller = None
            puck_speed_px_s = 0.0
            if puck_center is not None:
                if self.puck_prev_pos is not None:
                    d_px = euclidean(puck_center, self.puck_prev_pos)
                    puck_speed_px_s = d_px * fps
                # nearest player
                closest_id, min_dist = None, float('inf')
                for pid, info in frame_players.items():
                    d = euclidean(info["center"], puck_center)
                    if d < min_dist:
                        min_dist = d
                        closest_id = pid
                if (closest_id is not None) and (min_dist < puck_control_distance) and (self.puck_detection_count >= puck_frames):
                    current_controller = closest_id
                    self.player_stats[closest_id]["possession_frames"] += 1
                # shot detection
                if self.puck_prev_pos is not None:
                    speed_increase = puck_speed_px_s - getattr(self, "puck_prev_speed", 0.0)
                    if (current_controller is not None) and (speed_increase > shot_speed_inc) and (puck_speed_px_s > min_shot_speed_puck):
                        h,w = frame.shape[:2]
                        left_goal = (0, int(h*0.35), int(w*0.05), int(h*0.3))
                        right_goal = (int(w*0.95), int(h*0.35), int(w*0.05), int(h*0.3))
                        x_p,y_p = puck_center
                        on_target = False
                        lx,ly,lw,lh = left_goal
                        rx,ry,rw,rh = right_goal
                        if lx <= x_p <= lx+lw and ly <= y_p <= ly+lh:
                            on_target = True
                        if rx <= x_p <= rx+rw and ry <= y_p <= ry+rh:
                            on_target = True
                        # record shot
                        self.shot_events.append({"frame": frame_idx, "player_id": current_controller, "pos": puck_center, "speed_px_s": puck_speed_px_s, "on_target": on_target})
                        self.player_stats[current_controller]["shots"] += 1
                        if on_target:
                            self.player_stats[current_controller]["shots_on_target"] += 1
                # update puck prev
                self.puck_prev_pos = puck_center
                self.puck_prev_speed = puck_speed_px_s

            # touches counting
            if (self.prev_controller != current_controller) and (current_controller is not None) and (self.puck_detection_count >= puck_frames):
                self.player_stats[current_controller]["touches"] += 1
            self.prev_controller = current_controller

            if self.rink_coords is not None:
                try:
                    cv2.polylines(frame, [self.rink_coords.astype(int)], True, (0,255,0), 2)
                except Exception:
                    pass

            # players overlays
            for pid, info in frame_players.items():
                bbox = info["bbox"]
                speed_label = info["speed_label"]
                dist_label = info["dist_label"]
                poss_secs = self.player_stats[pid]["possession_frames"] / fps
                touches = self.player_stats[pid]["touches"]
                shots = self.player_stats[pid]["shots"]
                shots_on = self.player_stats[pid]["shots_on_target"]
                acc = (shots_on / shots * 100) if shots>0 else 0.0
                extra = f"P:{poss_secs:.1f}s T:{touches} Acc:{acc:.0f}%"
                self.player_overlay(frame, bbox, pid, speed_label, dist_label, extra)

            # puck draw 
            if puck_center is not None:
                cv2.circle(frame, puck_center, 6, puck_color, -1)

            
            self.dashboard(frame, frame_idx)

           
            self.frames_records.append({"frame": frame_idx, "players": frame_players, "puck": puck_center, "controller": current_controller})

            out.write(frame)
            if frame_idx % 200 == 0:
                print(f"[INFO] {frame_idx} frames processed...")

        cap.release()
        out.release()

        # save outputs
        os.makedirs(os.path.dirname(output_pickle) or ".", exist_ok=True)
        with open(output_pickle, "wb") as f:
            pickle.dump({"frames": self.frames_records, "player_stats": dict(self.player_stats), "shot_events": self.shot_events}, f)
        print("[INFO] Saved pickle:", output_pickle)

        # csv summary
        rows = []
        for pid,s in self.player_stats.items():
            shots = s.get("shots", 0)
            shots_on = s.get("shots_on_target", 0)
            rows.append({"player_id": pid, "distance_m": s.get("distance_m", 0.0), "possession_s": s.get("possession_frames",0)/fps, "touches": s.get("touches",0), "shots": shots, "shots_on_target": shots_on, "shot_accuracy_pct": (shots_on/shots*100) if shots>0 else 0.0})
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print("Saved CSV:", output_csv)
        print("Processing complete.")

    # drawing helpers
    def player_overlay(self, frame, bbox, track_id, speed_label, dist_label, extra_text=""):
        x1,y1,x2,y2 = map(int, bbox)
        x_center = (x1+x2)//2
        bottom_y = y2
        # ellipse
        try:
            cv2.ellipse(frame, (x_center, bottom_y), (int((x2-x1)//2), int(0.35*(x2-x1))), 0, -45, 235, ellipse_clor, 2, cv2.LINE_AA)
        except Exception:
            pass
        # id box
        rw,rh = 56,22
        rx1 = x_center - rw//2
        ry1 = bottom_y - rh - 6
        rx2 = rx1 + rw
        ry2 = ry1 + rh
        cv2.rectangle(frame, (rx1,ry1), (rx2,ry2), id_box, -1)
        cv2.putText(frame, f"ID:{track_id}", (rx1+6, ry1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_text, 1, cv2.LINE_AA)
        # speed & distance
        tx = x1
        ty_speed = y1 - 10
        ty_dist = y1 - 28
        if ty_speed < 12:
            ty_speed = y1 + 18
            ty_dist = y1 + 34
        cv2.putText(frame, speed_label, (tx, ty_speed), cv2.FONT_HERSHEY_PLAIN, 0.9, text, 2, cv2.LINE_AA)
        cv2.putText(frame, dist_label, (tx, ty_dist), cv2.FONT_HERSHEY_PLAIN, 0.9, text, 2, cv2.LINE_AA)
        
        if extra_text:
            et_y = ry1 - 8
            if et_y < 12:
                et_y = ry1 + rh + 18
            cv2.putText(frame, extra_text, (x1, et_y), cv2.FONT_HERSHEY_PLAIN, 0.8, text, 2, cv2.LINE_AA)

    def dashboard(self, frame, frame_idx):
        h,w = frame.shape[:2]
        overlay = frame.copy()
        alpha = 0.6
        x0,y0 = 8, h-140
        x1,y1 = 360, h-8
        cv2.rectangle(overlay, (x0,y0), (x1,y1), (200,200,200), -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        lines = [f"Frame: {frame_idx}", f"Players tracked: {len(self.cum_distance_px)}", f"Shots detected: {len(self.shot_events)}"]
        for i,line in enumerate(lines):
            cv2.putText(frame, line, (x0+8, y0+20+i*20), cv2.FONT_HERSHEY_PLAIN, 0.9, text, 1, cv2.LINE_AA)

# main
if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_pickle) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    analyzer = PlayerPuckAnalyzer(player_path, puck_path, rink_coords=RINK_COORDS)
    analyzer.process(input_video, output_video, frame_skip=FRAME_SKIP, downscale=DOWNSCALE)
