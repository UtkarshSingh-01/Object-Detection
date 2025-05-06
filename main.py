import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import cv2
from ultralytics import YOLO
import time

class YOLOv8App:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        self.root.geometry("850x650")
        self.root.configure(bg="#1e1e1e")  # Dark background

        self.model = YOLO('yolov8n.pt')  # YOLOv8 nano model
        self.cap = None
        self.running = False
        self.recording = False
        self.writer = None
        self.last_frame = None

        # Title label
        self.title_label = tk.Label(root, text="Object Detection", font=("Helvetica", 20, "bold"),
                                    bg="#1e1e1e", fg="white")
        self.title_label.pack(pady=10)

        # Video display area
        self.video_label = tk.Label(root, bg="#1e1e1e")
        self.video_label.pack(pady=10)

        # Buttons frame
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=20)

        self.start_btn = self.create_button(btn_frame, "Start Detection", self.start_detection, 0)
        self.stop_btn = self.create_button(btn_frame, "Stop", self.stop_detection, 1, tk.DISABLED)
        self.snapshot_btn = self.create_button(btn_frame, "Capture Snapshot", self.capture_snapshot, 2, tk.DISABLED)
        self.record_btn = self.create_button(btn_frame, "Start Recording", self.toggle_recording, 3, tk.DISABLED)

    def create_button(self, parent, text, command, column, state=tk.NORMAL):
        btn = tk.Button(parent, text=text, command=command, width=15, font=("Helvetica", 10, "bold"),
                        bg="#3c3c3c", fg="white", activebackground="#5c5c5c", activeforeground="white",
                        relief="flat", cursor="hand2", state=state)
        btn.grid(row=0, column=column, padx=10, pady=5)
        return btn

    def start_detection(self):
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.snapshot_btn.config(state=tk.NORMAL)
        self.record_btn.config(state=tk.NORMAL)
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self.detect_objects).start()

    def stop_detection(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.snapshot_btn.config(state=tk.DISABLED)
        self.record_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        self.video_label.config(image="")

    def toggle_recording(self):
        self.recording = not self.recording
        self.record_btn.config(text="Stop Recording" if self.recording else "Start Recording")
        if self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter("yolo_output.avi", fourcc, 20.0, (640, 480))
        else:
            if self.writer:
                self.writer.release()

    def capture_snapshot(self):
        if self.last_frame is not None:
            filename = f"yolo_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.last_frame)
            messagebox.showinfo("Snapshot", f"Saved as {filename}")

    def detect_objects(self):
        def update_frame():
            if not self.running or not self.cap or not self.cap.isOpened():
                if self.cap:
                    self.cap.release()
                if self.writer:
                    self.writer.release()
                return

            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                results = self.model(frame, verbose=False)[0]
                annotated_frame = results.plot()
                self.last_frame = annotated_frame.copy()

                if self.recording and self.writer:
                    self.writer.write(annotated_frame)

                img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(img_pil)

                self.video_label.imgtk = imgtk  # keep reference to avoid garbage collection
                self.video_label.configure(image=imgtk)

            self.root.after(30, update_frame)  # ~30 fps

        update_frame()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8App(root)
    root.mainloop()
