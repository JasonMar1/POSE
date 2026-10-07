"""Tkinter launcher: pick an exercise, run a workout session, repeat."""

import ctypes
import tkinter as tk
from tkinter import messagebox

import cv2

import rep_counter as rc
from overlay import THEME


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


def _enable_dpi_awareness():
    """Tell Windows this process handles its own scaling. Without this, Tk
    renders at a lower logical resolution and the OS upscales the result to
    fit the real screen -- which is what makes the window look blurry."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # fallback for older Windows
        except (AttributeError, OSError):
            pass  # not on Windows, or DPI awareness already set elsewhere


class App:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.device = None
        self.cap = None
        self.status_var = tk.StringVar(value="")

        root.title("POSE Gym App")
        root.configure(bg=_hex(THEME["card"]))
        root.resizable(False, False)
        self._center(440, 560)
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.loading_label = tk.Label(
            root, text="Loading model and camera...",
            bg=_hex(THEME["card"]), fg=_hex(THEME["sub"]), font=("Segoe UI", 13))
        self.loading_label.pack(expand=True)
        root.update()  # paint the loading label before the blocking model load

        try:
            self.model, self.device = rc.load_model()
        except Exception as e:
            self._show_load_error(str(e))
            return

        self.loading_label.destroy()
        self._build_picker()

    def _center(self, w, h):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x, y = (sw - w) // 2, (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _show_load_error(self, msg):
        self.loading_label.config(
            text=f"Could not load the pose model:\n{msg}",
            fg=_hex(THEME["bad"]), wraplength=380, justify="center")
        tk.Button(
            self.root, text="QUIT", command=self.quit,
            bg=_hex(THEME["card"]), fg=_hex(THEME["sub"]),
            activebackground=_hex(THEME["card"]), activeforeground=_hex(THEME["bad"]),
            relief="flat", bd=0, font=("Segoe UI", 11, "bold"), cursor="hand2",
        ).pack(pady=20)

    def _build_picker(self):
        card_bg = _hex(THEME["card"])
        container = tk.Frame(self.root, bg=card_bg)
        container.pack(expand=True, fill="both")

        tk.Label(container, text="POSE", bg=card_bg, fg=_hex(THEME["text"]),
                 font=("Segoe UI", 24, "bold")).pack(pady=(48, 4))
        tk.Label(container, text="Your Gym Assistant", bg=card_bg, fg=_hex(THEME["text"]),
                 font=("Segoe UI", 12, "bold")).pack(pady=(48, 4))
        tk.Label(container, text="Choose an exercise to begin", bg=card_bg,
                 fg=_hex(THEME["sub"]), font=("Segoe UI", 12)).pack(pady=(4, 28))

        for key, exercise in rc.EXERCISES.items():
            self._make_exercise_button(container, key, exercise["label"])

        tk.Frame(container, bg=_hex(THEME["track"]), height=1).pack(
            fill="x", padx=40, pady=20)

        quit_btn = tk.Button(
            container, text="QUIT", command=self.quit,
            bg=card_bg, fg=_hex(THEME["sub"]),
            activebackground=card_bg, activeforeground=_hex(THEME["bad"]),
            relief="flat", bd=0, font=("Segoe UI", 11, "bold"), cursor="hand2")
        quit_btn.pack(pady=(0, 8))
        quit_btn.bind("<Enter>", lambda e: quit_btn.config(fg=_hex(THEME["bad"])))
        quit_btn.bind("<Leave>", lambda e: quit_btn.config(fg=_hex(THEME["sub"])))

        tk.Label(container, textvariable=self.status_var, bg=card_bg,
                 fg=_hex(THEME["sub"]), font=("Segoe UI", 11)).pack(pady=(4, 0))

    def _make_exercise_button(self, parent, key, label):
        track, accent = _hex(THEME["track"]), _hex(THEME["accent"])
        btn = tk.Button(
            parent, text=label, command=lambda: self._start(key),
            bg=track, fg=_hex(THEME["text"]), activebackground=accent,
            activeforeground=_hex(THEME["text"]), relief="flat", bd=0,
            font=("Segoe UI", 13, "bold"), height=2, cursor="hand2")
        btn.pack(fill="x", padx=40, pady=8)
        btn.bind("<Enter>", lambda e: btn.config(bg=accent))
        btn.bind("<Leave>", lambda e: btn.config(bg=track))

    def _ensure_camera(self):
        if self.cap is not None and self.cap.isOpened():
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = rc.open_capture(0, 1280, 720, 30)

    def _start(self, exercise_key):
        try:
            self._ensure_camera()
        except RuntimeError as e:
            messagebox.showerror("Camera Error", str(e))
            return

        # Swap the picker for the OpenCV workout window. Both are fine sharing
        # one process: cv2's imshow/waitKey drive their own native window and
        # don't need Tk's mainloop, and this call is synchronous, so there's
        # no re-entrancy to worry about -- the picker just reappears after.
        self.root.withdraw()
        count = None
        try:
            count = rc.run_session(exercise_key, self.model, self.device, self.cap)
        except Exception as e:
            messagebox.showerror("Session Error", str(e))
        finally:
            self.root.deiconify()

        if count is not None:
            label = rc.EXERCISES[exercise_key]["label"].lower()
            self.status_var.set(f"Last session: {count} {label}")

    def quit(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


def main():
    _enable_dpi_awareness()  # must happen before the Tk window is created
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
