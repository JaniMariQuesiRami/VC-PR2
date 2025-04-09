import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import sys

# Inicializar modelo global de YOLO (se usará en todas las funciones)
modelo_yolo = YOLO("yolov8n.pt")

# Colores predefinidos en formato BGR para la detección de pose
color_options = {
    "Rojo": (0, 0, 255),
    "Verde": (0, 255, 0),
    "Azul": (255, 0, 0),
    "Amarillo": (0, 255, 255),
    "Cian": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Blanco": (255, 255, 255),
    "Naranja": (0, 128, 255)
}
pose_color = color_options["Verde"]  # Color inicial

# ---------------------------------------------------------------------------
# Sección 1: Funcionalidad de detección de pose (primer código)
# ---------------------------------------------------------------------------

def detectar_y_dibujar(frame, pose, draw_color):
    annotated = frame.copy()
    black = np.zeros_like(frame)
    results = modelo_yolo.predict(frame, classes=[0], verbose=False)[0]
    height, width = frame.shape[:2]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            for connection in mp.solutions.pose.POSE_CONNECTIONS:
                start = res.pose_landmarks.landmark[connection[0]]
                end = res.pose_landmarks.landmark[connection[1]]
                x_start = int(x1 + start.x * (x2 - x1))
                y_start = int(y1 + start.y * (y2 - y1))
                x_end = int(x1 + end.x * (x2 - x1))
                y_end = int(y1 + end.y * (y2 - y1))
                cv2.line(annotated, (x_start, y_start), (x_end, y_end), draw_color, 2)
                cv2.line(black, (x_start, y_start), (x_end, y_end), draw_color, 2)
    return annotated, black

def procesar_video_pose(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ No se pudo abrir el video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30

    print(f"🎞 Resolución: {width}x{height}, FPS: {fps}")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_overlay = cv2.VideoWriter("overlay_output.mp4", fourcc, fps, (width, height))
    out_black = cv2.VideoWriter("skeleton_output.mp4", fourcc, fps, (width, height))

    with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=0) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            overlay, black = detectar_y_dibujar(frame, pose, pose_color)
            cv2.imshow("Overlay (Video - Pose)", overlay)
            cv2.imshow("Skeleton (Video - Pose)", black)

            out_overlay.write(overlay)
            out_black.write(black)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out_overlay.release()
    out_black.release()
    cv2.destroyAllWindows()
    print("✅ Procesamiento de video finalizado.")

def procesar_camara_pose():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ No se pudo abrir la cámara.")
        return

    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            overlay, black = detectar_y_dibujar(frame, pose, pose_color)
            cv2.imshow("Overlay (Live - Pose)", overlay)
            cv2.imshow("Skeleton (Live - Pose)", black)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Transmisión de cámara detenida.")

def elegir_color():
    global pose_color

    def set_color(nombre):
        global pose_color
        pose_color = color_options[nombre]
        color_window.destroy()

    color_window = tk.Toplevel(root)
    color_window.title("Selecciona un color")
    tk.Label(color_window, text="Elige un color para los contornos:").pack(pady=10)
    for nombre in color_options:
        tk.Button(
            color_window,
            text=nombre,
            bg='#%02x%02x%02x' % tuple(reversed(color_options[nombre])),  # Conversión BGR a RGB
            width=20,
            command=lambda n=nombre: set_color(n)
        ).pack(pady=2)

def elegir_video(proceso):
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if filepath:
        proceso(filepath)

# ---------------------------------------------------------------------------
# Sección 2: Funcionalidades adicionales (detección de rostros, face mesh, manos, holistic)
# ---------------------------------------------------------------------------

def get_person_boxes(frame, conf_threshold=0.5):
    results = modelo_yolo(frame)
    boxes = []
    # Cada detección en formato [x1, y1, x2, y2, conf, cls]
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0 and conf >= conf_threshold:
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = int(x2)
            y2 = int(y2)
            boxes.append((x1, y1, x2, y2))
    return boxes

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

def process_holistic(crop):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = holistic.process(crop_rgb)
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

def run_detection_with_yolo(process_function, window_title):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ No se pudo abrir la cámara.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes = get_person_boxes(frame)
        for (x1, y1, x2, y2) in boxes:
            crop = frame[y1:y2, x1:x2].copy()
            processed_crop = process_function(crop)
            frame[y1:y2, x1:x2] = processed_crop
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(window_title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Ahora saldrá con "q"
            break
    cap.release()
    cv2.destroyAllWindows()

# Funciones para llamar a cada detección adicional desde la GUI
def detection_face():
    run_detection_with_yolo(process_face_detection, "Detección de Rostros")

def detection_face_mesh():
    run_detection_with_yolo(process_face_mesh, "Detección de Face Mesh")

def detection_hands():
    run_detection_with_yolo(process_hands, "Detección de Manos")

def detection_holistic():
    run_detection_with_yolo(process_holistic, "Detección Holística")

# ---------------------------------------------------------------------------
# Interfaz gráfica con Tkinter
# ---------------------------------------------------------------------------

root = tk.Tk()
root.title("Detector de Poses y Funcionalidades Adicionales con YOLO y MediaPipe")

tk.Label(root, text="Elige una fuente de entrada y funcionalidad").pack(pady=10)

# Sección de detección de pose
frame_pose = tk.LabelFrame(root, text="Detección de Pose")
frame_pose.pack(padx=10, pady=5, fill="both")

tk.Button(frame_pose, text="📹 Procesar video (Pose)", command=lambda: elegir_video(procesar_video_pose), width=30).pack(pady=5)
tk.Button(frame_pose, text="📷 Cámara en vivo (Pose)", command=procesar_camara_pose, width=30).pack(pady=5)
tk.Button(frame_pose, text="🎨 Cambiar color de contorno", command=elegir_color, width=30).pack(pady=5)

# Sección de otras detecciones basadas en YOLO + MediaPipe
frame_add = tk.LabelFrame(root, text="Otras Detecciones con YOLO y MediaPipe")
frame_add.pack(padx=10, pady=5, fill="both")

tk.Button(frame_add, text="👤 Detección de Rostros (Face Detection)", command=detection_face, width=40).pack(pady=5)
tk.Button(frame_add, text="🗺 Detección de Keypoints en Rostros (Face Mesh)", command=detection_face_mesh, width=40).pack(pady=5)
tk.Button(frame_add, text="✋ Detección de Manos (Hands)", command=detection_hands, width=40).pack(pady=5)
tk.Button(frame_add, text="💡 Detección Holística (Rostro, Pose y Manos)", command=detection_holistic, width=40).pack(pady=5)

tk.Button(root, text="❌ Cerrar programa", command=lambda: [root.quit(), root.destroy(), sys.exit()], width=30).pack(pady=15)

root.mainloop()
