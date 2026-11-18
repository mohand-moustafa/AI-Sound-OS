"""Modern desktop interface for the Sound Operating System."""
from __future__ import annotations

import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk
from typing import Optional

from commands.executor import execute
from nlp.classifier import IntentClassifier
from nlp.extractor import extract


BG = "#0b1020"
PANEL = "#121a2d"
PANEL_ALT = "#18233b"
CARD = "#1b2945"
TEXT = "#eef4ff"
MUTED = "#91a4c4"
ACCENT = "#6c7cff"
ACCENT_HOVER = "#8190ff"
SUCCESS = "#49d69a"
WARNING = "#ffc857"
ERROR = "#ff7182"


def process_command(text: str, classifier: Optional[IntentClassifier] = None,
                    dry_run: bool = False) -> dict:
    """Run one command through the same pipeline used by the terminal app."""
    classifier = classifier or IntentClassifier()
    intent, confidence = classifier.predict(text)
    params = extract(intent, text)
    result = execute(intent, params, dry_run=dry_run)
    return {
        "text": text,
        "intent": intent,
        "confidence": confidence,
        "params": params,
        "success": result.success,
        "message": result.message,
    }


class SoundOSApp(tk.Tk):
    """Desktop shell around the existing classifier, extractor, and executor."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Sound OS  •  Voice Control Center")
        self.geometry("1180x760")
        self.minsize(980, 650)
        self.configure(bg=BG)

        self.classifier = IntentClassifier()
        self.is_listening = False
        self.stop_event = threading.Event()
        self.wake_listener = None
        self.history: list[dict] = []
        self._setup_styles()
        self._build_ui()
        self._set_status("Ready to listen", SUCCESS)
        self.after(350, self._start_wake_mode)

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED,
                        font=("Segoe UI", 11))
        style.configure("CardTitle.TLabel", background=CARD, foreground=MUTED,
                        font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", background=CARD, foreground=TEXT,
                        font=("Segoe UI", 13, "bold"))
        style.configure("Accent.TButton", background=ACCENT, foreground="white",
                        borderwidth=0, padding=(18, 12), font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER)])
        style.configure("Secondary.TButton", background=PANEL_ALT, foreground=TEXT,
                        borderwidth=0, padding=(14, 10), font=("Segoe UI", 10))
        style.map("Secondary.TButton", background=[("active", CARD)])
        style.configure("TCheckbutton", background=BG, foreground=MUTED,
                        font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", BG)], foreground=[("active", TEXT)])

    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=28, pady=24)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(2, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 22))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Sound OS", style="Title.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(header, text="Your voice. Your Linux workspace. One command away.",
                  style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.status_label = ttk.Label(header, text="", style="Muted.TLabel")
        self.status_label.grid(row=0, column=1, rowspan=2, sticky="e")
        self.status_dot = tk.Canvas(header, width=12, height=12, bg=BG, highlightthickness=0)
        self.status_dot.grid(row=0, column=2, rowspan=2, padx=(10, 0))

        self._build_sidebar(root)
        self._build_workspace(root)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        sidebar = ttk.Frame(parent, style="Panel.TFrame", padding=20)
        sidebar.grid(row=2, column=1, sticky="nsew", padx=(22, 0))
        tk.Label(sidebar, text="CAPABILITIES", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 14))
        intents = [
            ("▸", "Open & close apps"),
            ("▣", "Create files & folders"),
            ("≡", "Browse Home folders"),
            ("◌", "System information"),
            ("◷", "Date & time"),
            ("◉", "Screenshot & OCR"),
        ]
        for icon, label in intents:
            row = tk.Frame(sidebar, bg=PANEL)
            row.pack(fill="x", pady=6)
            tk.Label(row, text=icon, bg=PANEL, fg=ACCENT,
                     font=("Segoe UI Symbol", 14)).pack(side="left", padx=(0, 10))
            tk.Label(row, text=label, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 10)).pack(side="left")

        tk.Frame(sidebar, bg=PANEL_ALT, height=1).pack(fill="x", pady=22)
        tk.Label(sidebar, text="QUICK COMMANDS", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 10))
        for command in ("open Chrome", "show system information",
                        "take a screenshot", "what time is it"):
            ttk.Button(sidebar, text=command, style="Secondary.TButton",
                       command=lambda value=command: self._fill_command(value)).pack(
                           fill="x", pady=4)

    def _build_workspace(self, parent: ttk.Frame) -> None:
        workspace = ttk.Frame(parent)
        workspace.grid(row=2, column=0, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(2, weight=1)

        command_card = ttk.Frame(workspace, style="Panel.TFrame", padding=20)
        command_card.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        command_card.columnconfigure(0, weight=1)
        tk.Label(command_card, text="COMMAND CENTER", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=3,
                                                    sticky="w", pady=(0, 12))
        self.command_var = tk.StringVar()
        self.command_entry = tk.Entry(
            command_card, textvariable=self.command_var, bg=PANEL_ALT, fg=TEXT,
            insertbackground=TEXT, relief="flat", font=("Segoe UI", 13), highlightthickness=1,
            highlightbackground=PANEL_ALT, highlightcolor=ACCENT)
        self.command_entry.grid(row=1, column=0, sticky="ew", ipady=12, padx=(0, 10))
        self.command_entry.bind("<Return>", lambda _event: self._run_typed_command())
        ttk.Button(command_card, text="Run command", style="Accent.TButton",
                   command=self._run_typed_command).grid(row=1, column=1, padx=(0, 8))
        self.listen_button = ttk.Button(command_card, text="■  Stop wake mode",
                                        style="Secondary.TButton", command=self._toggle_listening)
        self.listen_button.grid(row=1, column=2)
        self.dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(command_card, text="Safe preview (dry run)",
                        variable=self.dry_run_var).grid(row=2, column=0, columnspan=3,
                                                        sticky="w", pady=(12, 0))

        metrics = ttk.Frame(workspace)
        metrics.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        for index in range(3):
            metrics.columnconfigure(index, weight=1)
        self.intent_value = self._metric_card(metrics, 0, "LAST INTENT", "—")
        self.confidence_value = self._metric_card(metrics, 1, "CONFIDENCE", "—")
        self.activity_value = self._metric_card(metrics, 2, "ACTIVITY", "Ready")

        history_panel = ttk.Frame(workspace, style="Panel.TFrame", padding=20)
        history_panel.grid(row=2, column=0, sticky="nsew")
        history_panel.rowconfigure(1, weight=1)
        history_panel.columnconfigure(0, weight=1)
        tk.Label(history_panel, text="ACTIVITY LOG", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.log = tk.Text(history_panel, bg=PANEL, fg=TEXT, insertbackground=TEXT,
                           relief="flat", wrap="word", font=("Consolas", 10),
                           padx=4, pady=14, state="disabled")
        self.log.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.log.tag_configure("time", foreground=MUTED)
        self.log.tag_configure("success", foreground=SUCCESS)
        self.log.tag_configure("error", foreground=ERROR)
        self.log.tag_configure("accent", foreground=ACCENT_HOVER)

    def _metric_card(self, parent: ttk.Frame, column: int, title: str,
                     value: str) -> ttk.Label:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(16, 13))
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        label = ttk.Label(card, text=value, style="CardValue.TLabel")
        label.pack(anchor="w", pady=(7, 0))
        return label

    def _fill_command(self, command: str) -> None:
        self.command_var.set(command)
        self.command_entry.focus_set()

    def _set_status(self, message: str, color: str) -> None:
        self.status_label.configure(text=message)
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 10, 10, fill=color, outline=color)
        self.activity_value.configure(text=message)

    def _log(self, message: str, tag: str = "") -> None:
        self.log.configure(state="normal")
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{stamp}] ", "time")
        self.log.insert("end", f"{message}\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _run_typed_command(self) -> None:
        text = self.command_var.get().strip()
        if text:
            self.command_var.set("")
            self._run_command_async(text)

    def _run_command_async(self, text: str) -> None:
        self._set_status("Processing…", WARNING)
        self._log(f"Command: {text}", "accent")
        dry_run = self.dry_run_var.get()
        thread = threading.Thread(target=self._process_worker,
                                  args=(text, dry_run), daemon=True)
        thread.start()

    def _process_worker(self, text: str, dry_run: bool) -> None:
        try:
            result = process_command(text, self.classifier, dry_run)
        except Exception as error:  # surface unexpected runtime errors in the UI
            self.after(0, lambda: self._show_error(str(error)))
            return
        self.after(0, lambda: self._show_result(result))

    def _show_result(self, result: dict) -> None:
        self.history.append(result)
        self.intent_value.configure(text=result["intent"])
        self.confidence_value.configure(text=f'{result["confidence"]:.0%}')
        color = SUCCESS if result["success"] else ERROR
        self._set_status("Completed" if result["success"] else "Needs attention", color)
        self._log(f'{result["intent"]}  •  {result["message"]}',
                  "success" if result["success"] else "error")

    def _show_error(self, message: str) -> None:
        self._set_status("Error", ERROR)
        self._log(f"Error: {message}", "error")

    def _toggle_listening(self) -> None:
        if self.is_listening:
            self.stop_event.set()
            if self.wake_listener is not None:
                self.wake_listener.stop()
            return
        self._start_wake_mode()

    def _start_wake_mode(self) -> None:
        if self.is_listening:
            return
        self.is_listening = True
        self.stop_event.clear()
        self.listen_button.configure(text="■  Stop wake mode")
        self._set_status('Say "hey sound"', ACCENT_HOVER)
        self._log('Wake-word mode active. Say "hey sound", then your command.', "accent")
        threading.Thread(target=self._wake_worker, daemon=True).start()

    def _wake_worker(self) -> None:
        try:
            from speech.transcriber import transcribe
            from speech.wake_word import WakeWordListener

            def on_status(message: str) -> None:
                labels = {
                    "LOADING_MODEL": "Loading speech model…",
                    "WAITING_FOR_WAKE_WORD": 'Say "hey sound"',
                    "WAKE_WORD_DETECTED": "Wake word detected",
                    "LISTENING": "Listening for command…",
                    "NO_SPEECH_TIMEOUT": "No command heard",
                }
                label = labels.get(message, message.replace("_", " ").title())
                self.after(0, lambda value=label: self._set_status(value, ACCENT_HOVER))

            self.wake_listener = WakeWordListener(on_status=on_status)
            while not self.stop_event.is_set():
                audio = self.wake_listener.listen_for_command()
                if audio is None or self.stop_event.is_set():
                    break
                text = transcribe(audio)
                if text:
                    self.after(0, lambda value=text: self._run_command_async(value))
                else:
                    self.after(0, lambda: self._log("I couldn't understand that.", "error"))
            self.after(0, lambda: self._finish_wake_mode("Wake mode stopped"))
        except Exception as error:
            error_message = str(error)
            self.after(0, lambda: self._finish_wake_mode("Microphone unavailable"))
            self.after(0, lambda message=error_message: self._show_error(message))
        finally:
            self.wake_listener = None

    def _listen_worker(self) -> None:
        try:
            from speech.audio_capture import record_until_silence
            from speech.transcriber import transcribe

            def on_status(message: str) -> None:
                self.after(0, lambda: self._set_status(message.title(), ACCENT_HOVER))

            audio = record_until_silence(on_status=on_status,
                                         stop_check=self.stop_event.is_set)
            if self.stop_event.is_set():
                self.after(0, lambda: self._finish_listening("Listening stopped"))
                return
            if audio is None:
                self.after(0, lambda: self._finish_listening("No speech detected"))
                return
            text = transcribe(audio)
            self.after(0, lambda: self._finish_listening("Transcribed"))
            if text:
                self.after(0, lambda: self._run_command_async(text))
            else:
                self.after(0, lambda: self._log("I couldn't understand that.", "error"))
        except Exception as error:
            error_message = str(error)
            self.after(0, lambda: self._finish_listening("Microphone unavailable"))
            self.after(0, lambda message=error_message: self._show_error(message))

    def _finish_listening(self, status: str) -> None:
        self.is_listening = False
        self.stop_event.clear()
        self.listen_button.configure(text="▶  Start wake mode")
        self._set_status(status, SUCCESS if status == "Transcribed" else WARNING)

    def _finish_wake_mode(self, status: str) -> None:
        self.is_listening = False
        self.stop_event.clear()
        self.listen_button.configure(text="▶  Start wake mode")
        self._set_status(status, WARNING)


def launch() -> None:
    SoundOSApp().mainloop()
