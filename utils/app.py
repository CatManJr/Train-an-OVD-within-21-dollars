import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolo11s-earth.pt')  # Load your model

# Default classes
default_classes = [
    'Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter', 'Passenger Vehicle',
    'Small Car', 'Bus', 'Pickup Truck', 'Utility Truck', 'Truck', 'Cargo Truck',
    'Truck w/Box', 'Truck Tractor', 'Trailer', 'Truck w/Flatbed', 'Truck w/Liquid',
    'Crane Truck', 'Railway Vehicle', 'Passenger Car', 'Cargo Car', 'Flat Car',
    'Tank car', 'Locomotive', 'Maritime Vessel', 'Motorboat', 'Sailboat', 'Tugboat',
    'Barge', 'Fishing Vessel', 'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker',
    'Engineering Vehicle', 'Tower crane', 'Container Crane', 'Reach Stacker', 'Straddle Carrier',
    'Mobile Crane', 'Dump Truck', 'Haul Truck', 'Scraper/Tractor', 'Front loader/Bulldozer',
    'Excavator', 'Cement Mixer', 'Ground Grader', 'Hut/Tent', 'Shed', 'Building',
    'Aircraft Hangar', 'Damaged Building', 'Facility', 'Construction Site', 'Vehicle Lot',
    'Helipad', 'Storage Tank', 'Shipping container lot', 'Shipping Container', 'Pylon', 'Tower'
]

def process_frame(frame, classes_input):
    global model  # Declare model as global

    # Process user input classes
    if classes_input and classes_input.strip():
        classes_list = [cls.strip() for cls in classes_input.split(',')]
        # Validate classes_list
        for cls in classes_list:
            if not isinstance(cls, str):
                print("Invalid class name:", cls)
                continue
        
        # Reload model with new classes
        model = YOLO('yolo11s-earth.pt')  # Reload the base model
        model.set_classes(classes_list)  # Set model classes
        
        # Create a mapping from class index to class name
        class_name_mapping = {i: name for i, name in enumerate(classes_list)}
    else:
        # Use default classes if no input or input is empty
        model = YOLO('yolo11s-earth.pt')  # Reload the base model
        model.set_classes(default_classes)
        # Create a mapping from class index to class name
        class_name_mapping = {i: name for i, name in enumerate(default_classes)}
    
    # Copy frame to a writable array
    frame = frame.copy()
    
    # Resize image to speed up processing (optional)
    h, w = frame.shape[:2]
    new_size = (640, int(h * (640 / w))) if w > h else (int(w * (640 / h)), 640)
    resized_frame = cv2.resize(frame, new_size)
    
    # Convert image format
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Use model for detection
    results = model.predict(rgb_frame)
    
    # Draw detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            
            class_name = class_name_mapping.get(int(cls), "Unknown")  # Get class name from mapping
            
            # Adjust coordinates to original image size
            x1 = int(x1 * w / new_size[0])
            y1 = int(y1 * h / new_size[1])
            x2 = int(x2 * w / new_size[0])
            y2 = int(y2 * h / new_size[1])
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}:{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    return frame

def main():
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# YOLO11s-Earth open vocabulary detection (DIOR finetuning)")
        with gr.Row():
            cam_input = gr.Image(type="numpy", sources=["webcam"], streaming=True, label="Webcam")
            classes_input = gr.Textbox(label="New classes (comma-separated)", placeholder="e.g.: airplane, airport, tennis court")
        output = gr.Image(label="Results", type="numpy", height=480)  # Set height to 480
        
        cam_input.stream(
            process_frame,
            inputs=[cam_input, classes_input],
            outputs=output
        )
    
    # Launch Gradio app
    demo.launch()

if __name__ == "__main__":
    main()