# inference.py
from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import torch
import gc
import random
import glob

class AccidentDetector:
    def __init__(self, model_path='models/mustafa/best3.pt'):
        self.model = YOLO(model_path)
        self.base_path = "incidents"
        self.audio_dir = "src/assets/audio/"
        
        # Filter accident classes
        self.accident_classes = [
            name for name in self.model.names.values() 
            if name.endswith('_accident')
        ]
        
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.base_path, exist_ok=True)
        
    def clean_memory(self):  # Fixed spelling
        try:
            # Clear CUDA cache if GPU is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear CPU memory
            gc.collect()
        except Exception as e:
            print(f"Warning: Memory cleanup failed: {e}")
    
    def create_incident_folder(self, camera_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{camera_name}_{timestamp}"
        path = os.path.join(self.base_path, folder_name)
        os.makedirs(path, exist_ok=True)
        return path
        
    def get_random_alert_sound(self):
        """Returns random audio file path from assets/audio directory"""
        audio_files = glob.glob(os.path.join(self.audio_dir, "*.mp3"))
        if not audio_files:
            return None
        return random.choice(audio_files)
    
    def process_video(self, video_path, camera_name, conf_threshold=0.25):
        try:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Frame buffer for accident clips
            buffer_size = int(fps * 4)  # 4 seconds total (±2)
            frame_buffer = deque(maxlen=buffer_size)
            
            output_path = "temp_output.mp4"
            out = cv2.VideoWriter(output_path, 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, 
                                (width, height))
            
            accident_detections = []
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                    
                # Store frame in buffer
                frame_buffer.append(frame.copy())
                
                timestamp = datetime.now() - timedelta(seconds=(cap.get(cv2.CAP_PROP_FRAME_COUNT) - frame_count) / fps)
                frame_count += 1
                
                results = self.model.predict(frame, conf=conf_threshold)
                
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        class_name = results[0].names[cls_id]
                        
                        # Only store accident detections
                        if class_name in self.accident_classes:
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            accident_detections.append({
                                'class_name': class_name,
                                'confidence': conf,
                                'box': xyxy,
                                'frame': frame.copy(),
                                'timestamp': timestamp,
                                'frame_idx': frame_count
                            })
                
                # Only draw high confidence detections
                if accident_detections:
                    top_conf = np.percentile([d['confidence'] for d in accident_detections], 75)
                    results[0].boxes = [box for box in results[0].boxes if float(box.conf[0]) >= top_conf]
                
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
            
            cap.release()
            out.release()
            
            if accident_detections:
                accident_detections.sort(key=lambda x: x['confidence'], reverse=True)
                top_3 = accident_detections[:3]
                
                # Create incident folder
                incident_path = self.create_incident_folder(camera_name)
                
                # Save crops
                crops = []
                for i, det in enumerate(top_3):
                    x1, y1, x2, y2 = map(int, det['box'])
                    crop = det['frame'][y1:y2, x1:x2]
                    crop_path = os.path.join(incident_path, f"detection_{i}.jpg")
                    cv2.imwrite(crop_path, crop)
                    crops.append(crop_path)
                
                # Fill remaining crop slots if needed
                while len(crops) < 3:
                    crops.append(None)
                
                # Save incident clip around highest confidence detection
                main_detection = accident_detections[0]
                clip_path = os.path.join(incident_path, "incident_clip.mp4")
                
                clip_writer = cv2.VideoWriter(clip_path,
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            fps,
                                            (width, height))
                
                for frame in frame_buffer:
                    clip_writer.write(frame)
                clip_writer
                
                # Save metadata
                metadata = {
                    'camera_name': camera_name,
                    'accident_type': top_3[0]['class_name'],
                    'timestamp': top_3[0]['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    'confidence': top_3[0]['confidence'],
                    'fps': fps,
                    'resolution': f"{width}x{height}",
                    'detections': [{
                        'class_name': det['class_name'],
                        'confidence': det['confidence'],
                        'timestamp': det['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    } for det in top_3]
                }
                
                with open(os.path.join(incident_path, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                # Prepare alert message
                alert = f"⚠️ ALERT: Accident detected by camera {camera_name}!\n"
                alert += f"\nIncident recorded at: {metadata['timestamp']}"
                alert_audio = self.get_random_alert_sound()
                for i, det in enumerate(top_3):
                    alert += f"\n\nDetection {i+1}:"
                    alert += f"\n  Time: {det['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                    alert += f"\n  Confidence: {det['confidence']*100:.1f}%"

                return output_path, alert, crops, alert_audio
                
            return output_path, "", [None, None, None], None
        finally:
            # Cleanup resources
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()
            self.clean_memory()

    def enhance_crop(self, crop, target_size=512):
        """Enhance cropped accident images using bicubic interpolation"""
        if crop is None:
            return None
            
        # Get current dimensions
        h, w = crop.shape[:2]
        
        # Calculate scaling factor to reach target size while maintaining aspect ratio
        scale = target_size / max(h, w)
        
        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Upscale using bicubic interpolation
        enhanced = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return enhanced