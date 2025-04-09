import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import sys

# Inicializar modelos globales
modelo_yolo = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose

# Colores predefinidos en formato BGR
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

# Función para detectar y dibujar la pose
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

# Procesar archivo de video
def procesar_video(input_path):
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

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=0) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            overlay, black = detectar_y_dibujar(frame, pose, pose_color)

            cv2.imshow("Overlay (Video)", overlay)
            cv2.imshow("Skeleton (Video)", black)

            out_overlay.write(overlay)
            out_black.write(black)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out_overlay.release()
    out_black.release()
    cv2.destroyAllWindows()
    print("✅ Procesamiento de video finalizado.")

# Procesar cámara en vivo
def procesar_camara():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ No se pudo abrir la cámara.")
        return

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
    print("🛑 Transmisión detenida.")


# Selector de color por botones
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
            bg='#%02x%02x%02x' % tuple(reversed(color_options[nombre])),  # Convertir BGR a RGB para mostrar
            width=20,
            command=lambda n=nombre: set_color(n)
        ).pack(pady=2)

# Elegir video
def elegir_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if filepath:
        procesar_video(filepath)

# Cerrar la app
def cerrar_programa():
    root.quit()
    root.destroy()
    sys.exit()

# GUI
root = tk.Tk()
root.title("Detector de Poses Multipersona")

tk.Label(root, text="Elige una fuente de entrada").pack(pady=10)
tk.Button(root, text="📹 Procesar video", command=elegir_video, width=30).pack(pady=5)
tk.Button(root, text="📷 Usar cámara en vivo", command=procesar_camara, width=30).pack(pady=5)
tk.Button(root, text="🎨 Cambiar color de contorno", command=elegir_color, width=30).pack(pady=5)
tk.Button(root, text="❌ Cerrar programa", command=cerrar_programa, width=30).pack(pady=15)

root.mainloop()
