# using Python 3.9 '' Face Detection and Pose ''
import cv2
import mediapipe

mp_face_detection = mediapipe.solutions.face_detection
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_pose = mediapipe.solutions.pose

capture = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        while True:
            success, frame = capture.read()

            if success == False:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frameRGB)
            resultsPose = pose.process(frameRGB)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

            mp_drawing.draw_landmarks(frame, resultsPose.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())

            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

capture.release()
cv2.destroyAllWindows()