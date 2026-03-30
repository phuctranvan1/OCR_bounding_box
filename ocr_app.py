"""
OCR Bounding Box Application
A Python GUI application that uses multiple OCR engines (Tesseract and EasyOCR)
to detect and display bounding boxes around text in images.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import zipfile
import threading
import io
import copy

from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import pytesseract
import easyocr
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
SIDEBAR_BG = "#2b2b2b"
TOOLBAR_BG = "#3c3f41"
CANVAS_BG = "#1e1e1e"
TEXT_COLOR = "#ffffff"
ACCENT_COLOR = "#4a9eff"
BTN_BG = "#4c5052"
BTN_ACTIVE = "#5a6365"
HIGHLIGHT_BOX = "#ff4444"
SELECTED_BOX = "#00cc44"
BOX_ALPHA = 0.35


# ---------------------------------------------------------------------------
# OCR Engines
# ---------------------------------------------------------------------------
class TesseractEngine:
    """Wrapper around pytesseract."""

    name = "Tesseract"

    def detect(self, pil_image: Image.Image):
        """
        Returns a list of dicts:
            {"text": str, "x": int, "y": int, "w": int, "h": int, "conf": float}
        """
        # Use image_to_data which gives word-level bounding boxes + confidences
        data = pytesseract.image_to_data(
            pil_image,
            output_type=pytesseract.Output.DICT,
            config="--psm 11",
        )
        results = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if text and conf > 30:
                results.append(
                    {
                        "text": text,
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                        "conf": conf / 100.0,
                    }
                )
        return results


class EasyOCREngine:
    """Wrapper around EasyOCR."""

    name = "EasyOCR"
    _reader = None

    @classmethod
    def _get_reader(cls):
        if cls._reader is None:
            cls._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return cls._reader

    def detect(self, pil_image: Image.Image):
        """
        Returns a list of dicts:
            {"text": str, "x": int, "y": int, "w": int, "h": int, "conf": float}
        Converts polygon boxes from EasyOCR to axis-aligned rectangles.
        """
        reader = self._get_reader()
        img_np = np.array(pil_image.convert("RGB"))
        raw = reader.readtext(img_np)
        results = []
        for bbox, text, conf in raw:
            text = text.strip()
            if not text or conf < 0.3:
                continue
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x, y = int(min(xs)), int(min(ys))
            w = int(max(xs) - x)
            h = int(max(ys) - y)
            results.append(
                {"text": text, "x": x, "y": y, "w": w, "h": h, "conf": conf}
            )
        return results


def merge_results(results_list):
    """
    Merge results from multiple OCR engines using Non-Maximum Suppression style
    deduplication.  Boxes with IoU > 0.5 are merged (the one with higher
    confidence wins).
    """

    def iou(a, b):
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["w"], by1 + b["h"]
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / union if union > 0 else 0.0

    all_boxes = []
    for results in results_list:
        all_boxes.extend(copy.deepcopy(results))

    # Sort by confidence descending
    all_boxes.sort(key=lambda r: r["conf"], reverse=True)

    kept = []
    suppressed = [False] * len(all_boxes)
    for i, box in enumerate(all_boxes):
        if suppressed[i]:
            continue
        kept.append(box)
        for j in range(i + 1, len(all_boxes)):
            if not suppressed[j] and iou(box, all_boxes[j]) > 0.5:
                suppressed[j] = True

    return kept


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """Apply preprocessing to improve OCR accuracy."""
    img = pil_image.convert("RGB")
    # Mild sharpening
    img = img.filter(ImageFilter.SHARPEN)
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    return img


# ---------------------------------------------------------------------------
# Bounding-box editing dialog
# ---------------------------------------------------------------------------
class BoxEditDialog(tk.Toplevel):
    """A popup that lets users view/copy/edit the text of a bounding box."""

    def __init__(self, parent, text: str, on_save=None):
        super().__init__(parent)
        self.title("Bounding Box Text")
        self.resizable(True, True)
        self.grab_set()

        self.on_save = on_save
        self._build(text)
        self.update_idletasks()
        self.geometry(f"420x220+{parent.winfo_rootx()+60}+{parent.winfo_rooty()+60}")

    def _build(self, text):
        frm = tk.Frame(self, bg=SIDEBAR_BG)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(frm, text="Detected text:", bg=SIDEBAR_BG, fg=TEXT_COLOR,
                 font=("Helvetica", 10, "bold")).pack(anchor="w")

        self._txt = tk.Text(frm, height=5, wrap="word",
                            bg="#3c3f41", fg=TEXT_COLOR,
                            insertbackground=TEXT_COLOR, relief="flat",
                            font=("Helvetica", 11))
        self._txt.insert("1.0", text)
        self._txt.pack(fill="both", expand=True, pady=5)

        btn_frm = tk.Frame(frm, bg=SIDEBAR_BG)
        btn_frm.pack(fill="x")

        for label, cmd in [
            ("Copy", self._copy),
            ("Save", self._save),
            ("Close", self.destroy),
        ]:
            tk.Button(btn_frm, text=label, command=cmd,
                      bg=BTN_BG, fg=TEXT_COLOR, activebackground=BTN_ACTIVE,
                      relief="flat", padx=10, pady=4).pack(side="left", padx=4)

    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._txt.get("1.0", "end-1c"))
        messagebox.showinfo("Copied", "Text copied to clipboard.", parent=self)

    def _save(self):
        if self.on_save:
            self.on_save(self._txt.get("1.0", "end-1c"))
        self.destroy()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class OCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OCR Bounding Box")
        self.geometry("1280x800")
        self.minsize(900, 600)
        self.configure(bg=CANVAS_BG)

        # State
        self._image_paths: list[str] = []        # list of file paths shown in list
        self._zip_members: dict = {}             # path -> (zip_path, member_name)
        self._current_path: str | None = None
        self._pil_image: Image.Image | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self._display_scale: float = 1.0
        self._offset_x: int = 0
        self._offset_y: int = 0

        # OCR results: list of {text, x, y, w, h, conf}
        self._ocr_results: list[dict] = []
        # canvas item ids -> box index
        self._box_items: dict[int, int] = {}
        self._selected_box: int | None = None

        # Image edits
        self._brightness = 1.0
        self._contrast = 1.0
        self._sharpness = 1.0
        self._rotation = 0

        # OCR engine selection
        self._engine_var = tk.StringVar(value="Both (merged)")
        self._engines = {
            "Tesseract": TesseractEngine(),
            "EasyOCR": EasyOCREngine(),
        }

        self._build_ui()
        self._bind_events()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_menubar()

        # Root layout: left sidebar | center canvas | right panel
        root_pane = tk.PanedWindow(self, orient="horizontal",
                                   bg=CANVAS_BG, sashrelief="flat",
                                   sashwidth=4)
        root_pane.pack(fill="both", expand=True)

        self._sidebar = self._build_sidebar(root_pane)
        root_pane.add(self._sidebar, minsize=200, width=220)

        self._center = self._build_center(root_pane)
        root_pane.add(self._center, minsize=400)

        self._right = self._build_right_panel(root_pane)
        root_pane.add(self._right, minsize=230, width=270)

    # --- Menu bar ---------------------------------------------------------
    def _build_menubar(self):
        menubar = tk.Menu(self)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image…", command=self._load_file,
                              accelerator="Ctrl+O")
        file_menu.add_command(label="Open Folder…", command=self._load_folder,
                              accelerator="Ctrl+Shift+O")
        file_menu.add_command(label="Open ZIP…", command=self._load_zip)
        file_menu.add_separator()
        file_menu.add_command(label="Save Result Image…",
                              command=self._save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Copy All OCR Text",
                              command=self._copy_all_text)
        edit_menu.add_command(label="Clear Bounding Boxes",
                              command=self._clear_boxes)
        edit_menu.add_separator()
        edit_menu.add_command(label="Reset Image Edits",
                              command=self._reset_edits)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Zoom In", command=self._zoom_in,
                              accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self._zoom_out,
                              accelerator="Ctrl+-")
        view_menu.add_command(label="Fit to Window", command=self._fit_image,
                              accelerator="Ctrl+0")
        menubar.add_cascade(label="View", menu=view_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    # --- Left sidebar -----------------------------------------------------
    def _build_sidebar(self, parent):
        frame = tk.Frame(parent, bg=SIDEBAR_BG)

        # ---- Buttons ----
        btn_frame = tk.Frame(frame, bg=SIDEBAR_BG, pady=8)
        btn_frame.pack(fill="x", padx=8)

        tk.Label(btn_frame, text="⬛ Load Images", bg=SIDEBAR_BG,
                 fg=ACCENT_COLOR, font=("Helvetica", 11, "bold")).pack(
            anchor="w", pady=(0, 6))

        for label, cmd in [
            ("📄  Load Image File", self._load_file),
            ("📁  Load Image Folder", self._load_folder),
            ("🗜  Load ZIP File", self._load_zip),
        ]:
            tk.Button(btn_frame, text=label, command=cmd,
                      bg=BTN_BG, fg=TEXT_COLOR, activebackground=BTN_ACTIVE,
                      relief="flat", anchor="w", padx=8, pady=6,
                      font=("Helvetica", 10)).pack(fill="x", pady=2)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # ---- Image list ----
        tk.Label(frame, text="Images", bg=SIDEBAR_BG, fg=TEXT_COLOR,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=10)

        list_frame = tk.Frame(frame, bg=SIDEBAR_BG)
        list_frame.pack(fill="both", expand=True, padx=8, pady=4)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        self._img_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            bg="#3c3f41",
            fg=TEXT_COLOR,
            selectbackground=ACCENT_COLOR,
            selectforeground=TEXT_COLOR,
            relief="flat",
            activestyle="none",
            font=("Helvetica", 9),
        )
        scrollbar.config(command=self._img_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self._img_listbox.pack(side="left", fill="both", expand=True)

        self._img_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # ---- Directory path ----
        self._dir_var = tk.StringVar(value="No folder loaded")
        dir_lbl = tk.Label(frame, textvariable=self._dir_var,
                           bg=SIDEBAR_BG, fg="#888888",
                           wraplength=200, justify="left",
                           font=("Helvetica", 8))
        dir_lbl.pack(anchor="w", padx=8, pady=(4, 8))

        return frame

    # --- Center canvas ----------------------------------------------------
    def _build_center(self, parent):
        frame = tk.Frame(parent, bg=CANVAS_BG)

        # Toolbar above canvas
        toolbar = tk.Frame(frame, bg=TOOLBAR_BG, height=30)
        toolbar.pack(fill="x")

        self._status_var = tk.StringVar(value="No image loaded")
        tk.Label(toolbar, textvariable=self._status_var,
                 bg=TOOLBAR_BG, fg=TEXT_COLOR,
                 font=("Helvetica", 9)).pack(side="left", padx=10, pady=4)

        # Canvas with scrollbars
        canvas_frame = tk.Frame(frame, bg=CANVAS_BG)
        canvas_frame.pack(fill="both", expand=True)

        h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal")
        h_scroll.pack(side="bottom", fill="x")
        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical")
        v_scroll.pack(side="right", fill="y")

        self._canvas = tk.Canvas(
            canvas_frame,
            bg=CANVAS_BG,
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set,
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.pack(fill="both", expand=True)
        h_scroll.config(command=self._canvas.xview)
        v_scroll.config(command=self._canvas.yview)

        return frame

    # --- Right panel ------------------------------------------------------
    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=SIDEBAR_BG)

        # ---- OCR Engine selector ----
        eng_frame = tk.Frame(frame, bg=SIDEBAR_BG, pady=8)
        eng_frame.pack(fill="x", padx=10)

        tk.Label(eng_frame, text="OCR Engine", bg=SIDEBAR_BG, fg=ACCENT_COLOR,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")

        for opt in ["Tesseract", "EasyOCR", "Both (merged)"]:
            tk.Radiobutton(
                eng_frame, text=opt, variable=self._engine_var, value=opt,
                bg=SIDEBAR_BG, fg=TEXT_COLOR, selectcolor=BTN_BG,
                activebackground=SIDEBAR_BG, font=("Helvetica", 9),
            ).pack(anchor="w", pady=1)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # ---- OCR Detect button ----
        ocr_btn = tk.Button(
            frame,
            text="🔍  OCR Detect",
            command=self._run_ocr,
            bg=ACCENT_COLOR,
            fg=TEXT_COLOR,
            activebackground="#2a7fef",
            relief="flat",
            font=("Helvetica", 12, "bold"),
            pady=8,
            cursor="hand2",
        )
        ocr_btn.pack(fill="x", padx=10, pady=6)

        self._ocr_progress = ttk.Progressbar(frame, mode="indeterminate")
        self._ocr_progress.pack(fill="x", padx=10)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=8, pady=8)

        # ---- Image Adjustments ----
        tk.Label(frame, text="Image Adjustments", bg=SIDEBAR_BG,
                 fg=ACCENT_COLOR, font=("Helvetica", 11, "bold")).pack(
            anchor="w", padx=10)

        sliders = [
            ("Brightness", "_brightness", 0.5, 2.0, self._apply_edits),
            ("Contrast", "_contrast", 0.5, 2.0, self._apply_edits),
            ("Sharpness", "_sharpness", 0.0, 3.0, self._apply_edits),
        ]
        self._sliders: dict[str, tk.DoubleVar] = {}
        for label, attr, lo, hi, cmd in sliders:
            tk.Label(frame, text=label, bg=SIDEBAR_BG, fg=TEXT_COLOR,
                     font=("Helvetica", 9)).pack(anchor="w", padx=12)
            var = tk.DoubleVar(value=1.0)
            self._sliders[attr] = var
            s = tk.Scale(
                frame, variable=var, from_=lo, to=hi, resolution=0.05,
                orient="horizontal", bg=SIDEBAR_BG, fg=TEXT_COLOR,
                troughcolor=BTN_BG, activebackground=ACCENT_COLOR,
                highlightthickness=0, length=220,
                command=lambda _v, c=cmd: c(),
            )
            s.pack(padx=10, pady=2)

        # Rotation
        tk.Label(frame, text="Rotation (°)", bg=SIDEBAR_BG, fg=TEXT_COLOR,
                 font=("Helvetica", 9)).pack(anchor="w", padx=12)
        self._rot_var = tk.IntVar(value=0)
        rot_scale = tk.Scale(
            frame, variable=self._rot_var, from_=0, to=359, resolution=1,
            orient="horizontal", bg=SIDEBAR_BG, fg=TEXT_COLOR,
            troughcolor=BTN_BG, activebackground=ACCENT_COLOR,
            highlightthickness=0, length=220,
            command=lambda _v: self._apply_edits(),
        )
        rot_scale.pack(padx=10, pady=2)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # ---- Quick tools ----
        tk.Label(frame, text="Quick Tools", bg=SIDEBAR_BG, fg=ACCENT_COLOR,
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=10)

        for label, cmd in [
            ("↺  Rotate 90°", self._rotate90),
            ("⟳  Reset Edits", self._reset_edits),
            ("📋  Copy All Text", self._copy_all_text),
            ("💾  Save Result", self._save_result),
        ]:
            tk.Button(frame, text=label, command=cmd,
                      bg=BTN_BG, fg=TEXT_COLOR, activebackground=BTN_ACTIVE,
                      relief="flat", anchor="w", padx=8, pady=5,
                      font=("Helvetica", 10)).pack(fill="x", padx=10, pady=2)

        # ---- OCR results summary ----
        ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=8, pady=6)
        tk.Label(frame, text="Detected Text Summary", bg=SIDEBAR_BG,
                 fg=ACCENT_COLOR, font=("Helvetica", 10, "bold")).pack(
            anchor="w", padx=10)

        summary_frame = tk.Frame(frame, bg=SIDEBAR_BG)
        summary_frame.pack(fill="both", expand=True, padx=8, pady=4)

        sb = tk.Scrollbar(summary_frame, orient="vertical")
        self._summary_text = tk.Text(
            summary_frame,
            height=8,
            bg="#3c3f41",
            fg=TEXT_COLOR,
            relief="flat",
            font=("Helvetica", 8),
            wrap="word",
            yscrollcommand=sb.set,
            state="disabled",
        )
        sb.config(command=self._summary_text.yview)
        sb.pack(side="right", fill="y")
        self._summary_text.pack(side="left", fill="both", expand=True)

        return frame

    # ------------------------------------------------------------------
    # Event Binding
    # ------------------------------------------------------------------
    def _bind_events(self):
        self.bind("<Control-o>", lambda _e: self._load_file())
        self.bind("<Control-O>", lambda _e: self._load_folder())
        self.bind("<Control-plus>", lambda _e: self._zoom_in())
        self.bind("<Control-equal>", lambda _e: self._zoom_in())
        self.bind("<Control-minus>", lambda _e: self._zoom_out())
        self.bind("<Control-0>", lambda _e: self._fit_image())
        self._canvas.bind("<ButtonPress-1>", self._on_canvas_click)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Button-4>", self._on_mousewheel)
        self._canvas.bind("<Button-5>", self._on_mousewheel)

    # ------------------------------------------------------------------
    # Loading images
    # ------------------------------------------------------------------
    def _load_file(self):
        paths = filedialog.askopenfilenames(
            title="Open Image File(s)",
            filetypes=[
                ("Image files",
                 "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self._zip_members.clear()
        self._image_paths = list(paths)
        self._populate_list()
        self._dir_var.set(os.path.dirname(paths[0]))
        if self._image_paths:
            self._load_image(self._image_paths[0])

    def _load_folder(self):
        folder = filedialog.askdirectory(title="Open Image Folder")
        if not folder:
            return
        self._zip_members.clear()
        paths = []
        for name in sorted(os.listdir(folder)):
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(folder, name))
        if not paths:
            messagebox.showinfo("No images", "No supported image files found.")
            return
        self._image_paths = paths
        self._populate_list()
        self._dir_var.set(folder)
        self._load_image(self._image_paths[0])

    def _load_zip(self):
        zip_path = filedialog.askopenfilename(
            title="Open ZIP File",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
        )
        if not zip_path:
            return
        self._zip_members.clear()
        self._image_paths = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                ext = os.path.splitext(member)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    key = member
                    self._image_paths.append(key)
                    self._zip_members[key] = (zip_path, member)
        if not self._image_paths:
            messagebox.showinfo("No images", "No supported images in ZIP.")
            return
        self._populate_list()
        self._dir_var.set(f"ZIP: {zip_path}")
        self._load_image(self._image_paths[0])

    def _populate_list(self):
        self._img_listbox.delete(0, "end")
        for path in self._image_paths:
            self._img_listbox.insert("end", os.path.basename(path))
        if self._image_paths:
            self._img_listbox.selection_set(0)

    def _on_listbox_select(self, event=None):
        sel = self._img_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._load_image(self._image_paths[idx])

    def _load_image(self, path: str):
        """Load a PIL image from a file path or ZIP member key."""
        try:
            if path in self._zip_members:
                zip_path, member = self._zip_members[path]
                with zipfile.ZipFile(zip_path, "r") as zf:
                    data = zf.read(member)
                pil_img = Image.open(io.BytesIO(data)).convert("RGB")
            else:
                pil_img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot load image:\n{exc}")
            return

        self._current_path = path
        self._pil_image = pil_img
        self._rotation = 0
        self._rot_var.set(0)
        for var in self._sliders.values():
            var.set(1.0)
        self._brightness = self._contrast = self._sharpness = 1.0

        self._clear_boxes()
        self._render_image(pil_img)
        self._status_var.set(
            f"{os.path.basename(path)}  "
            f"({pil_img.width}×{pil_img.height})"
        )

    # ------------------------------------------------------------------
    # Image rendering
    # ------------------------------------------------------------------
    def _get_edited_image(self) -> Image.Image:
        """Return a copy of _pil_image with current edits applied."""
        if self._pil_image is None:
            return None
        img = self._pil_image.copy()
        # Rotation
        rot = self._rot_var.get()
        if rot:
            img = img.rotate(-rot, expand=True)
        # Brightness
        img = ImageEnhance.Brightness(img).enhance(
            self._sliders["_brightness"].get()
        )
        # Contrast
        img = ImageEnhance.Contrast(img).enhance(
            self._sliders["_contrast"].get()
        )
        # Sharpness
        img = ImageEnhance.Sharpness(img).enhance(
            self._sliders["_sharpness"].get()
        )
        return img

    def _render_image(self, pil_img: Image.Image, keep_zoom: bool = False):
        """Put the image on the canvas, optionally keeping the current zoom."""
        self._canvas.delete("all")
        self._box_items.clear()

        cw = self._canvas.winfo_width() or 800
        ch = self._canvas.winfo_height() or 600

        iw, ih = pil_img.size
        if not keep_zoom:
            # Fit to window
            scale = min(cw / iw, ch / ih, 1.0)
            self._display_scale = scale
        else:
            scale = self._display_scale

        nw, nh = int(iw * scale), int(ih * scale)
        self._offset_x = max(0, (cw - nw) // 2)
        self._offset_y = max(0, (ch - nh) // 2)

        resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(resized)
        self._canvas.create_image(
            self._offset_x, self._offset_y, anchor="nw", image=self._tk_image
        )
        self._canvas.config(scrollregion=(0, 0, nw + self._offset_x * 2,
                                          nh + self._offset_y * 2))

        # Re-draw any existing bounding boxes
        self._draw_boxes()

    def _apply_edits(self):
        if self._pil_image is None:
            return
        edited = self._get_edited_image()
        self._render_image(edited, keep_zoom=True)

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    def _zoom_in(self):
        self._display_scale *= 1.2
        self._apply_edits()

    def _zoom_out(self):
        self._display_scale = max(0.05, self._display_scale / 1.2)
        self._apply_edits()

    def _fit_image(self):
        if self._pil_image is None:
            return
        edited = self._get_edited_image()
        self._render_image(edited, keep_zoom=False)

    def _on_mousewheel(self, event):
        if event.delta > 0 or event.num == 4:
            self._zoom_in()
        else:
            self._zoom_out()

    # ------------------------------------------------------------------
    # Quick tools
    # ------------------------------------------------------------------
    def _rotate90(self):
        new_val = (self._rot_var.get() + 90) % 360
        self._rot_var.set(new_val)
        self._apply_edits()

    def _reset_edits(self):
        for var in self._sliders.values():
            var.set(1.0)
        self._rot_var.set(0)
        self._apply_edits()

    def _copy_all_text(self):
        if not self._ocr_results:
            messagebox.showinfo("No Results", "Run OCR detect first.")
            return
        text = "\n".join(r["text"] for r in self._ocr_results)
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copied", "All OCR text copied to clipboard.")

    def _clear_boxes(self):
        self._ocr_results = []
        self._box_items.clear()
        self._selected_box = None
        self._canvas.delete("bbox")
        self._update_summary([])

    def _save_result(self):
        if self._pil_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
        )
        if not path:
            return
        edited = self._get_edited_image()
        # Draw bounding boxes onto a copy of the image at full resolution.
        # OCR coordinates are in the edited image's pixel space, so no scaling needed.
        img_np = np.array(edited)
        for res in self._ocr_results:
            x, y, w, h = res["x"], res["y"], res["w"], res["h"]
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                img_np, res["text"][:30],
                (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                cv2.LINE_AA,
            )
        Image.fromarray(img_np).save(path)
        messagebox.showinfo("Saved", f"Result saved to:\n{path}")

    # ------------------------------------------------------------------
    # OCR Detection
    # ------------------------------------------------------------------
    def _run_ocr(self):
        if self._pil_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return

        self._clear_boxes()
        self._status_var.set("Running OCR…")
        self._ocr_progress.start(10)

        # Run in background thread to keep UI responsive
        thread = threading.Thread(target=self._ocr_worker, daemon=True)
        thread.start()

    def _ocr_worker(self):
        try:
            edited = self._get_edited_image()
            preprocessed = preprocess_image(edited)

            engine_choice = self._engine_var.get()
            results_list = []

            if engine_choice in ("Tesseract", "Both (merged)"):
                results_list.append(
                    self._engines["Tesseract"].detect(preprocessed)
                )
            if engine_choice in ("EasyOCR", "Both (merged)"):
                results_list.append(
                    self._engines["EasyOCR"].detect(preprocessed)
                )

            merged = merge_results(results_list)

            # Post back to main thread
            self.after(0, lambda: self._ocr_done(merged))
        except Exception as exc:
            self.after(
                0,
                lambda: (
                    messagebox.showerror("OCR Error", str(exc)),
                    self._status_var.set("OCR failed."),
                    self._ocr_progress.stop(),
                ),
            )

    def _ocr_done(self, results: list[dict]):
        self._ocr_progress.stop()
        self._ocr_results = results
        n = len(results)
        self._status_var.set(
            f"OCR complete — {n} text region{'s' if n != 1 else ''} found."
        )
        self._draw_boxes()
        self._update_summary(results)

    # ------------------------------------------------------------------
    # Bounding boxes
    # ------------------------------------------------------------------
    def _draw_boxes(self):
        self._canvas.delete("bbox")
        self._box_items.clear()

        scale = self._display_scale
        ox, oy = self._offset_x, self._offset_y

        for idx, res in enumerate(self._ocr_results):
            x1 = int(res["x"] * scale) + ox
            y1 = int(res["y"] * scale) + oy
            x2 = x1 + int(res["w"] * scale)
            y2 = y1 + int(res["h"] * scale)

            color = SELECTED_BOX if idx == self._selected_box else HIGHLIGHT_BOX

            rect = self._canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color,
                width=2,
                fill="",
                tags="bbox",
            )
            self._box_items[rect] = idx

    def _on_canvas_click(self, event):
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        # Find topmost bbox item at click position
        items = self._canvas.find_overlapping(cx - 1, cy - 1, cx + 1, cy + 1)
        for item in reversed(items):
            if item in self._box_items:
                idx = self._box_items[item]
                self._selected_box = idx
                self._draw_boxes()
                self._open_box_editor(idx)
                return

    def _open_box_editor(self, idx: int):
        res = self._ocr_results[idx]
        text = res["text"]

        def on_save(new_text):
            self._ocr_results[idx]["text"] = new_text
            self._update_summary(self._ocr_results)

        BoxEditDialog(self, text, on_save=on_save)

    def _update_summary(self, results: list[dict]):
        self._summary_text.config(state="normal")
        self._summary_text.delete("1.0", "end")
        if results:
            for i, r in enumerate(results):
                conf_pct = f"{r['conf']*100:.0f}%"
                self._summary_text.insert(
                    "end", f"[{i+1}] ({conf_pct}) {r['text']}\n"
                )
        else:
            self._summary_text.insert("end", "(no results)")
        self._summary_text.config(state="disabled")

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------
    def _show_about(self):
        messagebox.showinfo(
            "About",
            "OCR Bounding Box v1.0\n\n"
            "OCR Engines:\n"
            "  • Tesseract (via pytesseract)\n"
            "  • EasyOCR\n\n"
            "Click 'OCR Detect' to find text regions.\n"
            "Click any bounding box to view/edit the text.",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = OCRApp()
    app.mainloop()
