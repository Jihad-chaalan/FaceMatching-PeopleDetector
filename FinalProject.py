import cv2
import face_recognition
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO

# Global variables
reference_encoding = None
reference_image_path = "reference.jpg"
video_capture = None
running = False
yolo_model = YOLO("yolov8n.pt")
frame_counter = 0
current_person_count = 0

# GUI setup
window = tk.Tk()
window.title("Face Recognition with YOLOv8")
window.geometry("900x700")

# Image label for showing frames
video_label = tk.Label(window)
video_label.pack()

# Label for result
result_label = tk.Label(window, text="", font=("Arial", 24))
result_label.pack(pady=10)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    processed = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    return processed

def encode_face(image):
    processed = preprocess_image(image)
    encodings = face_recognition.face_encodings(processed)
    return encodings[0] if encodings else None

def capture_reference_photo():
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Info", "Press 'c' to capture reference photo")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Reference Photo - Press 'c'", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(reference_image_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Reference photo saved successfully.")

    global reference_encoding
    image = face_recognition.load_image_file(reference_image_path)
    reference_encoding = encode_face(image)

    if reference_encoding is not None:
        result_label.config(text="Reference photo captured", fg="green")
    else:
        result_label.config(text="No face detected in reference photo", fg="red")

def upload_reference_photo():
    global reference_encoding
    file_path = filedialog.askopenfilename(title="Select Reference Image",
                                           filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    image = face_recognition.load_image_file(file_path)
    reference_encoding = encode_face(image)

    if reference_encoding is not None:
        result_label.config(text="Reference photo uploaded", fg="green")
    else:
        result_label.config(text="No face detected in uploaded photo", fg="red")

def recognize_live():
    global running, video_capture
    if reference_encoding is None:
        messagebox.showerror("Error", "Please set a reference photo first!")
        return

    video_capture = cv2.VideoCapture(0)
    running = True
    update_frame()

def update_frame():
    global running, frame_counter, current_person_count
    if not running:
        return

    ret, frame = video_capture.read()
    if not ret:
        return

    label_text = ""
    label_color = "black"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)
    processed_frame = preprocess_image(small_frame)

    # --- Run YOLO every 5 frames to update person count ---
    if frame_counter % 5 == 0:
        results = yolo_model.predict(source=frame, conf=0.3, classes=[0], verbose=False)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            current_person_count = len(boxes)

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # --- Face Recognition if only 1 person detected ---
    if current_person_count == 1:
        face_locations = face_recognition.face_locations(processed_frame)
        face_encodings = face_recognition.face_encodings(processed_frame, face_locations)

        if face_encodings:
            face_encoding = face_encodings[0]
            face_distance = face_recognition.face_distance([reference_encoding], face_encoding)[0]
            match = face_recognition.compare_faces([reference_encoding], face_encoding)[0]

            if match and face_distance < 0.6:
                label_text = "Same Person"
                label_color = "green"
            else:
                label_text = "Different Person"
                label_color = "red"

            (top, right, bottom, left) = face_locations[0]
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(frame, label_text, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            label_text = "No face detected"
            label_color = "gray"

    elif current_person_count > 1:
        label_text = "More than one person detected!"
        label_color = "orange"
    elif current_person_count == 0:
        label_text = "No person detected"
        label_color = "gray"

    result_label.config(text=label_text, fg=label_color)

    # Update video display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    frame_counter += 1
    video_label.after(10, update_frame)

def stop_camera():
    global running, video_capture
    running = False
    if video_capture:
        video_capture.release()
        result_label.config(text="Camera stopped", fg="black")

# Buttons
btn_frame = tk.Frame(window)
btn_frame.pack(pady=20)

capture_btn = tk.Button(btn_frame, text="Capture Reference", command=capture_reference_photo, width=20, height=2)
capture_btn.grid(row=0, column=0, padx=10)

upload_btn = tk.Button(btn_frame, text="Upload Reference", command=upload_reference_photo, width=20, height=2)
upload_btn.grid(row=0, column=1, padx=10)

recognize_btn = tk.Button(btn_frame, text="Start Recognition", command=recognize_live, width=20, height=2)
recognize_btn.grid(row=0, column=2, padx=10)

stop_btn = tk.Button(btn_frame, text="Stop Camera", command=stop_camera, width=20, height=2)
stop_btn.grid(row=0, column=3, padx=10)

window.mainloop()
