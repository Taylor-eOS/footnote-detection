import os
import json
import tempfile
import tkinter as tk
from tkinter import messagebox
from settings import INPUT_FILE, OUTPUT_PATH, CONTEXT_GUI, CONTEXT_JSON

def load_entries(input_path=INPUT_FILE):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print("Cannot read", input_path, e)
        return []
    entries = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isdigit():
            j = i
            while j < n and text[j].isdigit():
                j += 1
            num = text[i:j]
            before_json = text[max(0, i - CONTEXT_JSON):i]
            after_json = text[j:j + CONTEXT_JSON]
            context = before_json + num + after_json
            before_display = text[max(0, i - CONTEXT_GUI):i]
            after_display = text[j:j + CONTEXT_GUI]
            display = (before_display + num + after_display).replace("\n", " ")
            while "  " in display:
                display = display.replace("  ", " ")
            display = display.strip()
            hs = display.find(num)
            he = hs + len(num)
            entries.append({
                "start": i,
                "end": j,
                "number": num,
                "context": context,
                "snippet": display,
                "highlight_start": hs,
                "highlight_end": he,
                "label": 0
            })
            i = j
        else:
            i += 1
    return entries

def atomic_write_json(path, data):
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix="tmp", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        try:
            os.remove(tmp)
        except:
            pass
        print("Save failed", e)

def save_entries(entries, path=OUTPUT_PATH):
    atomic_write_json(path, entries)

class Labeler(tk.Tk):
    def __init__(self, entries):
        super().__init__()
        self.entries = entries
        self.title("Number Labeler")
        self.geometry("900x700")
        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        frame = tk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.text = tk.Text(frame, font=("Consolas", 11), wrap=tk.WORD, cursor="arrow")
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text.tag_configure("context", foreground="#666666")
        self.text.tag_configure("num", foreground="#c0392b", font=("Consolas", 11, "bold"))
        self.text.tag_configure("labeled", background="#d4edda")
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=self.text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.config(yscrollcommand=scrollbar.set)
        self.line_to_index = {}
        for idx, entry in enumerate(self.entries):
            line = idx + 1
            self.line_to_index[line] = idx
            s = entry["snippet"]
            hs = entry["highlight_start"]
            he = entry["highlight_end"]
            self.text.insert(tk.END, s[:hs], "context")
            self.text.insert(tk.END, s[hs:he], "num")
            self.text.insert(tk.END, s[he:] + "\n", "context")
            if entry.get("label", 0) == 1:
                self.text.tag_add("labeled", f"{line}.0", f"{line}.end")
        self.text.bind("<Key>", lambda e: "break")
        self.text.bind("<Button-1>", self._on_click)
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(btn_frame, text="Save", command=self._save).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Quit", command=self._on_quit).pack(side=tk.RIGHT, padx=5)
        self.status = tk.Label(self, text="", anchor="w")
        self.status.pack(fill=tk.X, padx=10, pady=2)

    def _status_text(self):
        total = len(self.entries)
        labeled = sum(1 for e in self.entries if e.get("label", 0) == 1)
        return f"{total} numbers total, {labeled} labeled 1"

    def _refresh_status(self):
        self.status.config(text=self._status_text())

    def _find_forward_consecutive(self, anchor_index):
        result = []
        try:
            anchor_num = int(self.entries[anchor_index]["number"])
        except Exception:
            return result
        next_expected = anchor_num + 1
        search_start = anchor_index + 1
        while search_start < len(self.entries):
            found = False
            for i in range(search_start, len(self.entries)):
                try:
                    val = int(self.entries[i]["number"])
                except Exception:
                    continue
                if val == next_expected:
                    result.append(i)
                    next_expected += 1
                    search_start = i + 1
                    found = True
                    break
            if not found:
                break
        return result

    def _on_click(self, event):
        clicked_index = self.text.index(f"@{event.x},{event.y}")
        line_num = int(clicked_index.split(".")[0])
        if line_num not in self.line_to_index:
            return "break"
        anchor_idx = self.line_to_index[line_num]
        if self.entries[anchor_idx]["label"] == 1:
            indices = [anchor_idx] + self._find_forward_consecutive(anchor_idx)
            for idx in indices:
                if self.entries[idx]["label"] == 1:
                    self.entries[idx]["label"] = 0
                    self.text.tag_remove("labeled", f"{idx + 1}.0", f"{idx + 1}.end")
        else:
            indices = [anchor_idx] + self._find_forward_consecutive(anchor_idx)
            for idx in indices:
                if self.entries[idx]["label"] != 1:
                    self.entries[idx]["label"] = 1
                    self.text.tag_add("labeled", f"{idx + 1}.0", f"{idx + 1}.end")
        self._refresh_status()
        self._save()
        return "break"

    def _save(self):
        save_entries(self.entries, OUTPUT_PATH)

    def _on_quit(self):
        self._save()
        self.destroy()

def main():
    entries = load_entries(INPUT_FILE)
    if not entries:
        print("No entries found.")
        return
    app = Labeler(entries)
    app.mainloop()

if __name__ == "__main__":
    main()
