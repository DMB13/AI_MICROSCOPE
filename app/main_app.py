import os
import threading
from pathlib import Path
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import time
import datetime
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import inference
from model.db import get_db
from model.model_config import MODEL_INPUT_SIZE
from model import report as report_utils

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Microscope Workstation")
        self.geometry("900x600")

        # Sidebar - patient registration
        self.sidebar = ctk.CTkFrame(self, width=220)
        self.sidebar.pack(side="left", fill="y", padx=8, pady=8)

        self.patient_id = ctk.CTkEntry(self.sidebar, placeholder_text="Patient ID / Case No")
        self.patient_id.pack(pady=(16, 8), padx=12, fill="x")

        self.capture_btn = ctk.CTkButton(self.sidebar, text="Capture Image", command=self.capture_image)
        self.capture_btn.pack(pady=8, padx=12, fill="x")

        self.upload_btn = ctk.CTkButton(self.sidebar, text="Upload Image/Video", command=self.upload_media)
        self.upload_btn.pack(pady=4, padx=12, fill="x")

        # Camera controls
        self.camera_menu = ctk.CTkOptionMenu(self.sidebar, values=["Detecting..."], command=self._on_camera_select)
        self.camera_menu.pack(pady=(8, 4), padx=12, fill="x")

        self.start_cam_btn = ctk.CTkButton(self.sidebar, text="Start Live", command=self.start_camera)
        self.start_cam_btn.pack(pady=4, padx=12, fill="x")

        self.stop_cam_btn = ctk.CTkButton(self.sidebar, text="Stop Live", command=self.stop_camera)
        self.stop_cam_btn.pack(pady=4, padx=12, fill="x")

        self.analyze_btn = ctk.CTkButton(self.sidebar, text="Run AI Diagnosis", command=self.run_diagnosis)
        self.analyze_btn.pack(pady=8, padx=12, fill="x")

        self.export_btn = ctk.CTkButton(self.sidebar, text="Export Reports", command=self.export_reports)
        self.export_btn.pack(pady=4, padx=12, fill="x")

        # Main area - image and results
        self.main_area = ctk.CTkFrame(self)
        self.main_area.pack(side="right", expand=True, fill="both", padx=8, pady=8)

        self.image_label = ctk.CTkLabel(self.main_area, text="No image captured")
        self.image_label.pack(pady=12)

        self.result_label = ctk.CTkLabel(self.main_area, text="Result: —")
        self.result_label.pack(pady=6)

        self.gradcam_label = ctk.CTkLabel(self.main_area, text="Grad-CAM")
        self.gradcam_label.pack(pady=6)

        self.captured_image_path = None
        self.model = None
        # camera state
        self.cap = None
        self.camera_running = False
        self.current_frame = None

        # populate camera list (non-blocking)
        self.after(100, self._populate_cameras)
        # load settings (export directory, etc.)
        try:
            settings_path = Path(__file__).resolve().parent / 'microscope_settings.json'
            if settings_path.exists():
                with settings_path.open('r', encoding='utf-8') as fh:
                    self.settings = json.load(fh)
            else:
                self.settings = {}
        except Exception:
            self.settings = {}

    def capture_image(self):
        # if live camera is running, capture current frame
        if self.camera_running and self.current_frame is not None:
            self.capture_from_camera()
            return

        path = filedialog.askopenfilename(title="Select image to simulate capture",
                                          filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        self.captured_image_path = path
        img = Image.open(path).resize((512, 512))
        self.tkimg = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tkimg, text="")
        self.result_label.configure(text="Result: waiting")

    def upload_media(self):
        """Allow user to upload an image or a video file. For videos, extract a single frame."""
        path = filedialog.askopenfilename(title="Select image or video",
                                          filetypes=[
                                              ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                                              ("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.avi")
                                          ])
        if not path:
            return

        suffix = Path(path).suffix.lower()
        image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

        save_dir = Path(__file__).resolve().parents[1] / "model" / "records"
        save_dir.mkdir(parents=True, exist_ok=True)

        if suffix in image_exts:
            # simple image upload
            try:
                dest = save_dir / Path(path).name
                from shutil import copy2
                copy2(path, dest)
                self.captured_image_path = str(dest)
                img = Image.open(dest).resize((512, 512))
                self.tkimg = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.tkimg, text="")
                self.result_label.configure(text="Uploaded image")
            except Exception as e:
                messagebox.showerror("Upload error", f"Failed to upload image: {e}")
        elif suffix in video_exts:
            # extract a frame from video (middle frame)
            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("Unable to open video file")
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                # choose middle frame or first if unknown
                target = frame_count // 2 if frame_count > 0 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError("Failed to read frame from video")
                timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                out = save_dir / f"upload_video_frame_{timestamp}.png"
                cv2.imwrite(str(out), frame)
                self.captured_image_path = str(out)
                img = Image.open(out).resize((512, 512))
                self.tkimg = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.tkimg, text="")
                self.result_label.configure(text="Uploaded video (frame extracted)")
            except Exception as e:
                messagebox.showerror("Upload error", f"Failed to extract frame: {e}")
        else:
            messagebox.showwarning("Unsupported file", "Selected file type is not supported.")

    def _populate_cameras(self):
        # Try to detect available camera indices 0-5
        cams = []
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            except Exception:
                cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                cams.append(str(i))
                cap.release()
        if not cams:
            cams = ["No camera detected"]
        self.camera_menu.configure(values=cams)
        # select first available
        try:
            self.camera_menu.set(cams[0])
        except Exception:
            pass

    def _on_camera_select(self, value):
        # noop; selection read at start
        return

    def run_diagnosis(self):
        if not self.captured_image_path:
            messagebox.showwarning("No image", "Please capture or select an image first.")
            return
        self.result_label.configure(text="Result: running...")
        thread = threading.Thread(target=self._diagnosis_worker, daemon=True)
        thread.start()

    def _diagnosis_worker(self):
        # Lazy load model
        if self.model is None:
            try:
                self.model = inference.load_model()
            except Exception as e:
                self.result_label.configure(text=f"Model load error: {e}")
                return

        try:
            prediction = inference.predict(self.captured_image_path, model=self.model)
        except Exception as e:
            self.result_label.configure(text=f"Prediction error: {e}")
            return

        species = prediction.get("species", "Unknown")
        conf = float(prediction.get("confidence", 0.0))
        self.result_label.configure(text=f"Result: {species} ({conf*100:.1f}%)")

        # Grad-CAM image (may be PIL.Image or None)
        gc_img = None
        try:
            gc_img = inference.grad_cam(self.captured_image_path, model=self.model)
        except Exception:
            gc_img = None

        if gc_img is not None:
            img = gc_img.resize((512, 512))
            self.tkgrad = ImageTk.PhotoImage(img)
            self.gradcam_label.configure(image=self.tkgrad, text="")

        # Persist record to SQLite DB
        try:
            db = get_db()
            # ensure model directory exists for saved images
            save_dir = Path(__file__).resolve().parents[1] / "model" / "records"
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            img_dest = save_dir / f"capture_{timestamp}.png"
            gradcam_dest = None

            # copy original captured image into records folder
            try:
                from shutil import copy2
                copy2(self.captured_image_path, img_dest)
            except Exception:
                img_dest = Path(self.captured_image_path)

            # save gradcam image if present
            if gc_img is not None:
                try:
                    gradcam_dest = save_dir / f"gradcam_{timestamp}.png"
                    gc_img.save(gradcam_dest)
                except Exception:
                    gradcam_dest = None

            db.insert_record(
                patient_id=(self.patient_id.get() or ""),
                species=species,
                confidence=conf,
                image_path=str(img_dest),
                gradcam_path=(str(gradcam_dest) if gradcam_dest is not None else None),
                timestamp=datetime.datetime.utcnow().isoformat()
            )
        except Exception as e:
            # log to UI but do not crash
            self.result_label.configure(text=f"Result saved error: {e}")

    # Camera control methods
    def start_camera(self):
        selected = self.camera_menu.get() if hasattr(self.camera_menu, 'get') else None
        cam_index = None
        try:
            cam_index = int(selected)
        except Exception:
            cam_index = 0

        # open capture
        if self.cap is None or not self.cap.isOpened():
            try:
                self.cap = cv2.VideoCapture(cam_index)
            except Exception:
                self.cap = cv2.VideoCapture(cam_index)

        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Unable to open camera {cam_index}")
            return

        self.camera_running = True
        self.result_label.configure(text="Live: running")
        self._update_frame()

    def stop_camera(self):
        self.camera_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.result_label.configure(text="Live: stopped")

    def _update_frame(self):
        if not self.camera_running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            # try again shortly
            self.after(100, self._update_frame)
            return
        # store current frame (BGR)
        self.current_frame = frame.copy()
        # convert to PIL and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((512, 512))
        self.tkimg = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tkimg, text="")
        # schedule next frame
        self.after(30, self._update_frame)

    def capture_from_camera(self):
        if self.current_frame is None:
            messagebox.showwarning("No frame", "No camera frame available to capture.")
            return
        save_dir = Path(__file__).resolve().parents[1] / "model" / "records"
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        img_dest = save_dir / f"capture_{timestamp}.png"
        # write BGR frame
        try:
            cv2.imwrite(str(img_dest), self.current_frame)
            self.captured_image_path = str(img_dest)
            img = Image.open(img_dest).resize((512, 512))
            self.tkimg = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.tkimg, text="")
            self.result_label.configure(text="Captured from camera")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save capture: {e}")

    def export_reports(self):
        """Export recent clinical records. User can choose Save As or use default export dir from settings."""
        try:
            db = get_db()
        except Exception as e:
            messagebox.showerror("DB error", f"Database not available: {e}")
            return

        # Ask user whether to use default export dir
        use_default = False
        default_dir = None
        try:
            default_dir = self.settings.get('export_dir') if hasattr(self, 'settings') else None
            if default_dir:
                use_default = messagebox.askyesno("Export location", f"Export to default directory?\n{default_dir}")
        except Exception:
            use_default = False

        if use_default and default_dir:
            out_dir = (Path(__file__).resolve().parents[1] / Path(default_dir)).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            # Default to CSV export in default directory, and also create a PDF summary
            csv_path = out_dir / f"clinical_export_{timestamp}.csv"
            pdf_path = out_dir / f"clinical_export_{timestamp}.pdf"
            try:
                db.export_csv(str(csv_path))
                try:
                    report_utils.export_recent_pdf(db=db, out_path=str(pdf_path))
                except Exception:
                    # Non-fatal: PDF may fail if reportlab not installed or thumbnails missing
                    pass
                messagebox.showinfo("Export", f"Exported {csv_path}\nAlso attempted PDF: {pdf_path}")
            except Exception as e:
                messagebox.showerror("Export error", f"Failed to export: {e}")
            return

        # Otherwise prompt Save As
        save_path = filedialog.asksaveasfilename(title="Save export as", defaultextension='.csv', filetypes=[('CSV','*.csv'), ('PDF','*.pdf')])
        if not save_path:
            return
        try:
            suffix = Path(save_path).suffix.lower()
            if suffix == '.pdf':
                try:
                    report_utils.export_recent_pdf(db=db, out_path=save_path)
                    messagebox.showinfo("Export", f"Exported {save_path}")
                except Exception as e:
                    messagebox.showerror("Export error", f"Failed to export PDF: {e}")
            else:
                db.export_csv(save_path)
                messagebox.showinfo("Export", f"Exported {save_path}")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to export: {e}")


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
