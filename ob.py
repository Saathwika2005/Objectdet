from ultralytics import YOLO

model = YOLO("yolov5x.pt")
results = model(source=0, save=True, stream=True) 
object_to_track = input("Enter the object name you want to track: ").lower()

matching_classes = [cls for cls in available_classes.values() if cls.lower() == object_to_track]

if not matching_classes:
    print("Invalid object name. Please choose from the available classes.")
else:
    
    for r in results:
        
        frame = r.orig_img

        for box in r.boxes:
            
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0]) 
            cls = int(box.cls) 
            
            detected_class_name = available_classes[cls]

            if detected_class_name.lower() == object_to_track:
    
                centroid_x = (x_min + x_max) // 2
                centroid_y = (y_min + y_max) // 2
               
                rgb_value = frame[centroid_y, centroid_x]

                print(f"Object: {detected_class_name}")
                print(f"Bounding Box: ({x_min}, {y_min}, {x_max}, {y_max})")
                print(f"Centroid: ({centroid_x}, {centroid_y})")
                print(f"RGB Value at Centroid: {rgb_value}")
 
