import ctypes
import tkinter as tk
from tkinter import messagebox, ttk

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
        self.camera_names = []
        self.camera_var = tk.StringVar()
        self.status_var = tk.StringVar(value="")
        self.pending_key = None                       # exercise chosen, waiting on goal screen
        self.goal_var = tk.StringVar(value="10")      # the reps box
        self.goal_title_var = tk.StringVar()          # exercise name shown on the goal screen
        self.goal_msg_var = tk.StringVar(value="")    # inline message on the goal screen

        root.title("POSE Gym App")
        root.configure(bg=_hex(THEME["card"]))
        root.resizable(False, False)
        self._center(440, 560)
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.loading_label = tk.Label(
            root, text="Loading model...",
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

        # Size the window to its content (fonts grow with DPI scaling, so a
        # hardcoded height can clip the bottom), then re-center it.
        self.root.update_idletasks()
        self._center(max(440, self.root.winfo_reqwidth()), self.root.winfo_reqheight())

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
        container = self.picker_frame = tk.Frame(self.root, bg=card_bg)
        container.pack(expand=True, fill="both")

        tk.Label(container, text="POSE", bg=card_bg, fg=_hex(THEME["text"]),
                 font=("Segoe UI", 24, "bold")).pack(pady=(48, 4))
        tk.Label(container, text="Your Gym Assistant", bg=card_bg, fg=_hex(THEME["text"]),
                 font=("Segoe UI", 12, "bold")).pack(pady=(48, 4))
        tk.Label(container, text="Choose an exercise to begin", bg=card_bg,
                 fg=_hex(THEME["sub"]), font=("Segoe UI", 12)).pack(pady=(4, 16))

        self._build_camera_selector(container, card_bg)

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

        self._build_goal_screen()   # built now, but only packed when an exercise is chosen

    def _build_camera_selector(self, parent, card_bg):
        track, text = _hex(THEME["track"]), _hex(THEME["text"])
        accent, sub = _hex(THEME["accent"]), _hex(THEME["sub"])

        # Dark-theme the dropdown. 'clam' is the ttk theme that honours custom
        # colours; the popup list is a plain Tk listbox styled via option_add.
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Cam.TCombobox", fieldbackground=track, background=track,
                        foreground=text, arrowcolor=text, bordercolor=track,
                        lightcolor=track, darkcolor=track, padding=6)
        style.map("Cam.TCombobox",
                  fieldbackground=[("readonly", track)],
                  foreground=[("readonly", text)],
                  selectbackground=[("readonly", track)],
                  selectforeground=[("readonly", text)])
        self.root.option_add("*TCombobox*Listbox.background", track)
        self.root.option_add("*TCombobox*Listbox.foreground", text)
        self.root.option_add("*TCombobox*Listbox.selectBackground", accent)
        self.root.option_add("*TCombobox*Listbox.selectForeground", text)
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 11))

        tk.Label(parent, text="CAMERA", bg=card_bg, fg=sub,
                 font=("Segoe UI", 9, "bold")).pack(padx=40, anchor="w")

        row = tk.Frame(parent, bg=card_bg)
        row.pack(fill="x", padx=40, pady=(2, 12))

        self.camera_box = ttk.Combobox(
            row, textvariable=self.camera_var, state="readonly",
            style="Cam.TCombobox", font=("Segoe UI", 11))
        self.camera_box.pack(side="left", fill="x", expand=True)
        self.camera_box.bind("<<ComboboxSelected>>",
                             lambda e: self.camera_box.selection_clear())

        refresh = tk.Button(
            row, text="Refresh", command=self._refresh_cameras,
            bg=card_bg, fg=sub, activebackground=card_bg, activeforeground=accent,
            relief="flat", bd=0, font=("Segoe UI", 10, "bold"), cursor="hand2")
        refresh.pack(side="left", padx=(10, 0))
        refresh.bind("<Enter>", lambda e: refresh.config(fg=accent))
        refresh.bind("<Leave>", lambda e: refresh.config(fg=sub))

        self._refresh_cameras()

    def _refresh_cameras(self):
        """Re-read the connected cameras, keeping the current choice if it's
        still plugged in."""
        previous = self.camera_var.get()
        self.camera_names = rc.list_cameras() or ["Camera 0"]
        self.camera_box["values"] = self.camera_names
        if previous in self.camera_names:
            self.camera_box.current(self.camera_names.index(previous))
        else:
            self.camera_box.current(0)

    def _make_exercise_button(self, parent, key, label):
        # `key` is a parameter of this method, so each button's lambda gets its
        # own copy (a lambda written directly inside the for-loop would share one).
        self._big_button(parent, label, lambda: self._choose_exercise(key))

    def _big_button(self, parent, text, command, bg=None, hover=None):
        """A full-width button with a hover colour; used on both screens."""
        bg = bg or _hex(THEME["track"])
        hover = hover or _hex(THEME["accent"])
        btn = tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=_hex(THEME["text"]), activebackground=hover,
            activeforeground=_hex(THEME["text"]), relief="flat", bd=0,
            font=("Segoe UI", 13, "bold"), height=2, cursor="hand2")
        btn.pack(fill="x", padx=40, pady=8)
        btn.bind("<Enter>", lambda e: btn.config(bg=hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    # ---- goal screen: shown after an exercise is chosen -------------------

    def _build_goal_screen(self):
        card_bg = _hex(THEME["card"])
        text, sub = _hex(THEME["text"]), _hex(THEME["sub"])
        track, accent = _hex(THEME["track"]), _hex(THEME["accent"])
        accent_hover = _hex(tuple(min(255, c + 30) for c in THEME["accent"]))

        f = self.goal_frame = tk.Frame(self.root, bg=card_bg)   # not packed yet

        tk.Label(f, textvariable=self.goal_title_var, bg=card_bg, fg=text,
                 font=("Segoe UI", 24, "bold")).pack(pady=(110, 4))
        tk.Label(f, text="How do you want to train?", bg=card_bg, fg=sub,
                 font=("Segoe UI", 12)).pack(pady=(0, 36))

        self._big_button(f, "FREE WORKOUT", self._start_free)

        tk.Label(f, text="OR SET A GOAL", bg=card_bg, fg=sub,
                 font=("Segoe UI", 9, "bold")).pack(pady=(32, 8))

        # Only let digits (max 3) be typed. Tk calls this before accepting each
        # keystroke, with the text the box WOULD contain ("%P"); False rejects it.
        digits_only = self.root.register(
            lambda proposed: proposed == "" or (proposed.isdigit() and len(proposed) <= 3))
        self.goal_spin = tk.Spinbox(
            f, from_=1, to=999, textvariable=self.goal_var, width=5, justify="center",
            font=("Segoe UI", 26, "bold"), bg=track, fg=text, insertbackground=text,
            buttonbackground=track, relief="flat", bd=0, highlightthickness=0,
            validate="key", validatecommand=(digits_only, "%P"))
        self.goal_spin.pack(ipady=6)
        self.goal_spin.bind("<Return>", lambda e: self._start_with_goal())
        tk.Label(f, text="reps", bg=card_bg, fg=sub,
                 font=("Segoe UI", 11)).pack(pady=(4, 0))

        self.goal_msg_label = tk.Label(f, textvariable=self.goal_msg_var, bg=card_bg,
                                       fg=sub, font=("Segoe UI", 10))
        self.goal_msg_label.pack(pady=(8, 8))

        self._big_button(f, "START WITH GOAL", self._start_with_goal,
                         bg=accent, hover=accent_hover)

        back = tk.Button(f, text="BACK", command=self._show_picker, bg=card_bg, fg=sub,
                         activebackground=card_bg, activeforeground=accent,
                         relief="flat", bd=0, font=("Segoe UI", 11, "bold"), cursor="hand2")
        back.pack(pady=(24, 0))
        back.bind("<Enter>", lambda e: back.config(fg=accent))
        back.bind("<Leave>", lambda e: back.config(fg=sub))

    def _goal_message(self, message, error=False):
        self.goal_msg_var.set(message)
        self.goal_msg_label.config(fg=_hex(THEME["bad" if error else "sub"]))

    def _choose_exercise(self, key):
        """Swap the home page for the goal screen."""
        self.pending_key = key
        self.goal_title_var.set(rc.EXERCISES[key]["label"])
        self._goal_message("")
        self.picker_frame.pack_forget()
        self.goal_frame.pack(expand=True, fill="both")
        self.goal_spin.focus_set()
        self.goal_spin.selection_range(0, "end")

    def _show_picker(self):
        self.goal_frame.pack_forget()
        self.picker_frame.pack(expand=True, fill="both")

    def _start_free(self):
        self._start(self.pending_key)

    def _start_with_goal(self):
        text = self.goal_var.get().strip()
        if not text.isdigit() or not 1 <= int(text) <= 999:
            self._goal_message("Enter a whole number from 1 to 999", error=True)
            return
        self._start(self.pending_key, goal=int(text))

    # ---- running a workout ------------------------------------------------

    def _start(self, exercise_key, goal=None):
        camera_index = self.camera_box.current()

        self._goal_message("Opening camera...")
        self.root.update_idletasks()
        try:
            cap = rc.open_capture(camera_index, 1280, 720, 30)
        except RuntimeError as e:
            self._goal_message("")
            messagebox.showerror("Camera Error", str(e))
            return          # stay on the goal screen so they can retry

        self._goal_message("")
        self._show_picker()   # so the home page is what reappears afterwards
        self.root.withdraw()
        result = None
        try:
            result = rc.run_session(exercise_key, self.model, self.device, cap, goal=goal)
        except Exception as e:
            messagebox.showerror("Session Error", str(e))
        finally:
            cap.release()
            self.root.deiconify()

        label = rc.EXERCISES[exercise_key]["label"].lower()
        if result is None:
            self.status_var.set("")
        elif result.goal_reached:
            self.status_var.set(f"Goal reached: {result.count}/{goal} {label}")
        elif goal is not None:
            self.status_var.set(f"Stopped at {result.count}/{goal} {label}")
        else:
            self.status_var.set(f"Last session: {result.count} {label}")

    def quit(self):
        self.root.destroy()


def main():
    _enable_dpi_awareness()  # must happen before the Tk window is created
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
