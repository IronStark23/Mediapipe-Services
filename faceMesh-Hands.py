# using Python 3.9 ''Face Mesh and Hands ''
import cv2
import mediapipe

mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_holistic = mediapipe.solutions.holistic

capture = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while True:
        success, frame = capture.read()

        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frameRGB)

        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None,
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec = None,
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Mesh - Hands', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()