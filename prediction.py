from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11n model
model = YOLO("/home/student/Desktop/spilkova/runs/detect/model_A8/weights/best.pt")

# Define path to video file
source = "/home/student/Desktop/spilkova/vidz/vid1.MOV"
cap = cv2.VideoCapture(source)

save_vid = True
if save_vid:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output3.MOV', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

if cap.isOpened():
    print('je to good')
else:
    print('ses pica baro')

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Write the processed frame to the output video
        if save_vid:
            out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()