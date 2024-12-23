# Incidents

Stores detected accident incidents:

```
incidents/
└── CAMERA-ID_TIMESTAMP/
    ├── incident_clip.mp4
    ├── metadata.json
    ├── detection_0.jpg
    ├── detection_1.jpg
    └── detection_2.jpg
```

## Metadata Format
```json
{
    "camera_name": "CCTV-001",
    "timestamp": "2024-01-01 12:00:00",
    "confidence": 0.95,
    "fps": 30,
    "resolution": "1920x1080"
}
```