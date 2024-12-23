import gradio as gr
from inference import AccidentDetector
from pathlib import Path
import os
import cv2

def process_with_params(video, camera_name):
    detector = AccidentDetector()
    # Use fixed low confidence threshold of 0.25
    video_output, alert, crops = detector.process_video(video, camera_name, 0.25)
    
    # Process and enhance crops
    enhanced_crops = []
    for crop_path in crops:
        if crop_path is not None:
            # Read image
            img = cv2.imread(crop_path)
            # Enhance
            enhanced = detector.enhance_crop(img)
            # Save back
            cv2.imwrite(crop_path, enhanced)
            enhanced_crops.append(crop_path)
        else:
            enhanced_crops.append(None)
    
    # Unpack enhanced crops
    crop1 = enhanced_crops[0] if enhanced_crops and len(enhanced_crops) > 0 else None
    crop2 = enhanced_crops[1] if enhanced_crops and len(enhanced_crops) > 1 else None
    crop3 = enhanced_crops[2] if enhanced_crops and len(enhanced_crops) > 2 else None
    
    return video_output, alert, crop1, crop2, crop3

def main():
    os.makedirs("temp", exist_ok=True)
    
    with gr.Blocks(title="Accident Detection System") as iface:
        gr.Markdown("# CCTV Accident Detection System")
        
        with gr.Row():
            with gr.Column():
                camera_name = gr.Textbox(
                    label="Camera Location/Name",
                    placeholder="e.g. Interstate-95 North Mile 67",
                    value="CCTV-001"
                )
                video_input = gr.Video(label="Input Video")
                
        with gr.Row():
            with gr.Column():
                video_output = gr.Video(label="Processed Video")
                alert_output = gr.Textbox(
                    label="Emergency Alert",
                    interactive=False,
                    lines=10
                )
                gr.Markdown("### Accident Images")
                with gr.Row():
                    crop_1 = gr.Image(label="Highest Confidence")
                    crop_2 = gr.Image(label="Second Highest")
                    crop_3 = gr.Image(label="Third Highest")
        
        submit_btn = gr.Button("Process Video")
        submit_btn.click(
            fn=process_with_params,
            inputs=[video_input, camera_name],
            outputs=[video_output, alert_output, crop_1, crop_2, crop_3]
        )
    
    try:
        iface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
    except Exception as e:
        print(f"Error launching interface: {e}")
        iface.launch()

if __name__ == "__main__":
    main()