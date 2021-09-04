# Advanced face detection script
import mediapipe as mp
import cv2

# Initialize mediapipe drawing and holistic modules
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic 


def main():
    # Grab a reference to the webcam
    camera = cv2.VideoCapture(0)
    # Initiate the holistic model
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as det:
        
        # Capture frame by frame
        while camera.isOpened():
            ret, frame = camera.read() # ret is true if frame is read correctly
            if not ret:
                print("[INFO]: Lost frame")
                pass

            # Change color of the frame
            frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect faces from the frame
            faces = det.process(frame_)
            # Change color back to BGR for rendering
            frame_ = cv2.cvtColor(frame_, cv2.COLOR_RGB2BGR)
            
            # Draw the face detected landmarks
            if faces is not None:
                mp_drawing.draw_landmarks(frame_, faces.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color = (0, 0, 255), thickness = 1, circle_radius = 1),
                    mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 1, circle_radius = 0.5)
                )

            # Draw a windows showing the frame
            cv2.imshow("Holistic face detection", frame_)

            # Check for a keypress
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    camera.release()
    cv2.destroyAllWindows()

# Run the main function if this is the main script
if __name__ == "__main__":
    main()
