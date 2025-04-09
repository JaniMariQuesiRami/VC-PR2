import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, colorchooser

# Inicializar modelos globales
modelo_yolo = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Color de líneas (editable)
pose_color = (0, 255, 0)

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
            for connection in mp_pose.POSE_CONNECTIONS:
                start = res.pose_landmarks.landmark[connection[0]]
                end = res.pose_landmarks.landmark[connection[1]]

                x_start = int(x1 + start.x * (x2 - x1))
                y_start = int(y1 + start.y * (y2 - y1))
                x_end = int(x1 + end.x * (x2 - x1))
                y_end = int(y1 + end.y * (y2 - y1))

                cv2.line(annotated, (x_start, y_start), (x_end, y_end), draw_color, 2)
                cv2.line(black, (x_start, y_start), (x_end, y_end), draw_color, 2)

    return annotated, black

def procesar_video(input_path):
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_overlay = cv2.VideoWriter("overlay_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out_black = cv2.VideoWriter("skeleton_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=0) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            overlay, black = detectar_y_dibujar(frame, pose, pose_color)

            cv2.imshow("Overlay", overlay)
            cv2.imshow("Skeleton", black)

            out_overlay.write(overlay)
            out_black.write(black)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out_overlay.release()
    out_black.release()
    cv2.destroyAllWindows()

def procesar_camara():
    cv2.startWindowThread()  # 🛠️ solución para macOS GUI
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            overlay, black = detectar_y_dibujar(frame, pose, pose_color)

            cv2.imshow("Overlay (Live)", overlay)
            cv2.imshow("Skeleton (Live)", black)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# GUI
def elegir_color():
    global pose_color
    rgb, _ = colorchooser.askcolor(title="Elegir color para los landmarks")
    if rgb:
        pose_color = tuple(int(c) for c in rgb)

def elegir_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if filepath:
        import threading
        threading.Thread(target=procesar_video, args=(filepath,), daemon=True).start()

# Crear GUI
root = tk.Tk()
root.title("Detector de Poses Multipersona")

tk.Label(root, text="Elige una fuente de entrada").pack(pady=10)
tk.Button(root, text="📹 Procesar video", command=elegir_video, width=30).pack(pady=5)
tk.Button(root, text="📷 Cámara en vivo", command=procesar_camara, width=30).pack(pady=5)
tk.Button(root, text="🎨 Cambiar color de contorno", command=elegir_color, width=30).pack(pady=10)

root.mainloop()
