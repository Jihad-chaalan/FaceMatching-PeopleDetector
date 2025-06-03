import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import os
from deepface import DeepFace

# Global variables
reference_image_path = "reference.jpg"
video_capture = None
running = False
reference_set = False

# GUI setup
window = tk.Tk()
window.title("Face Recognition App (DeepFace)")
window.geometry("900x700")

# Image label for showing frames
video_label = tk.Label(window)
video_label.pack()

# Label for result
result_label = tk.Label(window, text="", font=("Arial", 24))
result_label.pack(pady=10)

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

    global reference_set
    if os.path.exists(reference_image_path):
        reference_set = True
        result_label.config(text="Reference photo captured", fg="green")
    else:
        result_label.config(text="Failed to capture reference photo", fg="red")

def upload_reference_photo():
    global reference_set
    file_path = filedialog.askopenfilename(title="Select Reference Image",
                                           filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    try:
        img = cv2.imread(file_path)
        if img is not None:
            cv2.imwrite(reference_image_path, img)
            reference_set = True
            result_label.config(text="Reference photo uploaded", fg="green")
        else:
            result_label.config(text="Invalid image selected", fg="red")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to upload image: {str(e)}")

def recognize_live():
    global running, video_capture, reference_set
    if not reference_set:
        messagebox.showerror("Error", "Please set a reference photo first!")
        return

    video_capture = cv2.VideoCapture(0)
    running = True
    update_frame()

def update_frame():
    global running
    if not running:
        return

    ret, frame = video_capture.read()
    if not ret:
        return

    label_text = ""
    label_color = "black"

    temp_path = "live_frame.jpg"
    cv2.imwrite(temp_path, frame)

    try:
        result = DeepFace.verify(img1_path=reference_image_path,
                                 img2_path=temp_path,
                                 model_name="Facenet",
                                 enforce_detection=False)

        if result["verified"]:
            label_text = "Same Person"
            label_color = "green"
            color = (0, 255, 0)
        else:
            label_text = "Different Person"
            label_color = "red"
            color = (0, 0, 255)
    except Exception as e:
        label_text = "Error in recognition"
        label_color = "orange"
        color = (0, 165, 255)
        print("DeepFace Error:", str(e))

    result_label.config(text=label_text, fg=label_color)

    # Display frame with label
    frame = cv2.putText(frame, label_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

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
