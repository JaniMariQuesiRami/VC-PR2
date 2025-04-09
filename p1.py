import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Helper: obtiene las cajas de detección de personas usando YOLO
def get_person_boxes(frame, yolo_model, conf_threshold=0.5):
    results = yolo_model(frame)
    boxes = []
    # results[0].boxes.data es una lista de detecciones en formato [x1, y1, x2, y2, conf, cls]
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        # Suponiendo que la clase "person" es la 0 en COCO
        if int(cls) == 0 and conf >= conf_threshold:
            # Aseguramos valores enteros y que no se salgan de la imagen
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = int(x2)
            y2 = int(y2)
            boxes.append((x1, y1, x2, y2))
    return boxes

# Función para aplicar MediaPipe Face Detection en una región (crop)
def process_face_detection(crop):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_detection.process(crop_rgb)
        if results.detections:
            ch, cw, _ = crop.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                cx = int(bbox.xmin * cw)
                cy = int(bbox.ymin * ch)
                cw_box = int(bbox.width * cw)
                ch_box = int(bbox.height * ch)
                cv2.rectangle(crop, (cx, cy), (cx+cw_box, cy+ch_box), (255, 0, 0), 2)
    return crop

# Función para aplicar MediaPipe Face Mesh en una región (crop)
def process_face_mesh(crop):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=2,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(crop_rgb)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=crop,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )
    return crop

# Función para aplicar MediaPipe Hands en una región (crop)
def process_hands(crop):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = hands.process(crop_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=crop,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS
                )
    return crop

# Función para aplicar MediaPipe Holistic en una región (crop)
def process_holistic(crop):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = holistic.process(crop_rgb)
        # Dibuja malla facial, pose y manos
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image=crop,
                landmark_list=results.face_landmarks,
                connections=mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=crop,
                landmark_list=results.pose_landmarks,
                connections=mp_holistic.POSE_CONNECTIONS
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=crop,
                landmark_list=results.left_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=crop,
                landmark_list=results.right_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS
            )
    return crop

# Función que integra YOLO para detectar múltiples personas y aplica el procesamiento MediaPipe
def run_detection_with_yolo(process_function, window_title):
    # Carga el modelo YOLO de Ultralytics (modelo ligero: yolov8n)
    yolo_model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Obtén las cajas de las personas detectadas por YOLO
        boxes = get_person_boxes(frame, yolo_model)
        # Procesa cada región detectada
        for (x1, y1, x2, y2) in boxes:
            # Extraer la región de interés (ROI) con la persona
            crop = frame[y1:y2, x1:x2].copy()
            # Aplica la función de MediaPipe (por ejemplo, face, face mesh, hands o holistic)
            processed_crop = process_function(crop)
            # Reemplaza la región original del frame por el ROI procesado
            frame[y1:y2, x1:x2] = processed_crop
            # Dibuja la caja de YOLO en verde (opcional)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(window_title, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Salir con la tecla Esc
            break
    cap.release()
    cv2.destroyAllWindows()

# Menú interactivo para elegir la funcionalidad deseada
def show_menu():
    print("\nSeleccione la opción de detección con YOLO y MediaPipe:")
    print("1. Detección de Rostros (Face Detection)")
    print("2. Detección de Keypoints en Rostros (Face Mesh)")
    print("3. Detección de Manos (Hands)")
    print("4. Detección Holistic (Rostro, Pose y Manos)")
    option = input("Ingrese el número de la opción deseada: ")
    return option

if __name__ == "__main__":
    while True:
        choice = show_menu()
        if choice == "1":
            run_detection_with_yolo(process_face_detection, "Face Detection")
        elif choice == "2":
            run_detection_with_yolo(process_face_mesh, "Face Mesh")
        elif choice == "3":
            run_detection_with_yolo(process_hands, "Hands Detection")
        elif choice == "4":
            run_detection_with_yolo(process_holistic, "Holistic Detection")
        else:
            print("Opción no válida.")
        repeat = input("¿Desea seleccionar otra opción? (s/n): ")
        if repeat.lower() != "s":
            break
