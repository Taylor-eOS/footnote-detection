import json
import tkinter as tk
from tkinter import ttk
import os

def load_or_create_contexts(input_path="input.txt", json_path="number_contexts.json"):
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                contexts = json.load(f)
            print(f"Loaded {len(contexts)} existing labelled entries from {json_path}")
            return contexts
        except Exception as e:
            print(f"JSON load failed: {e}, creating new from input.txt")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except Exception as e:
        print(f"Cannot read input.txt: {e}")
        return []
    
    tokens = full_text.split()
    contexts = []
    
    for i, token in enumerate(tokens):
        cleaned = token.strip('.,;:!?()[]{}"\'')
        if not cleaned:
            continue
        try:
            float(cleaned)
        except ValueError:
            continue
            
        start = max(0, i - 20)
        end = min(len(tokens), i + 1 + 20)
        before = tokens[start:i]
        after = tokens[i + 1:end]
        context_parts = before + [cleaned] + after
        input_text = ' '.join(context_parts)
        
        contexts.append({
            'input_text': input_text,
            'number': cleaned,
            'label': 0
        })
    
    print(f"Created {len(contexts)} new entries from input.txt")
    return contexts


def save_contexts(contexts, json_path="number_contexts.json"):
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(contexts)} entries to {json_path}")
    except Exception as e:
        print(f"Save failed: {e}")


class Labeler:
    def __init__(self, contexts, json_path="number_contexts.json"):
        self.contexts = contexts
        self.json_path = json_path
        self.page_size = 8
        self.current_page = 0
        self.total_pages = (len(contexts) + self.page_size - 1) // self.page_size
        
        self.root = tk.Tk()
        self.root.title("Fast Binary Labeller (0/1) – Space = next page")
        self.root.geometry("1100x700")
        
        self.labels = [ctx['label'] for ctx in contexts]
        
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.frame)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.status = ttk.Label(self.root, text="")
        self.status.pack(pady=5)
        
        self.root.bind("<space>", self.next_page)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.show_page()
        self.root.mainloop()
    
    def show_page(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        start = self.current_page * self.page_size
        end = min(start + self.page_size, len(self.contexts))
        
        for idx in range(start, end):
            i = idx
            ctx = self.contexts[i]
            row_frame = ttk.Frame(self.scrollable_frame)
            row_frame.pack(fill=tk.X, pady=4)
            
            var = tk.IntVar(value=self.labels[i])
            chk = ttk.Checkbutton(
                row_frame,
                variable=var,
                command=lambda pos=i, v=var: self.toggle_label(pos, v)
            )
            chk.pack(side=tk.LEFT, padx=5)
            
            number_str = f"  {ctx['number']}  "
            text = f"{ctx['input_text']}"
            label_text = tk.Label(
                row_frame,
                text=number_str + text,
                anchor="w",
                justify=tk.LEFT,
                font=("TkDefaultFont", 10)
            )
            label_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            if self.labels[i] == 1:
                label_text.configure(background="#d4edda", fg="#155724")
            else:
                label_text.configure(background="#f8d7da", fg="#721c24")
            
            label_text.bind("<Button-1>", lambda e, pos=i, v=var: self.toggle_via_click(pos, v))
        
        status_text = f"Page {self.current_page + 1}/{self.total_pages}    "
        status_text += f"Entries {start+1}–{end} of {len(self.contexts)}    "
        status_text += f"Press SPACE to continue    Click/toggle to set 1"
        self.status.config(text=status_text)
    
    def toggle_label(self, pos, var):
        self.labels[pos] = var.get()
        self.contexts[pos]['label'] = var.get()
        save_contexts(self.contexts, self.json_path)
    
    def toggle_via_click(self, pos, var):
        new_val = 1 if var.get() == 0 else 0
        var.set(new_val)
        self.toggle_label(pos, var)
    
    def next_page(self, event=None):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.show_page()
        else:
            save_contexts(self.contexts, self.json_path)
            print("Reached last page – all changes saved")
            self.status.config(text="Finished labelling – window can be closed")
    
    def on_closing(self):
        save_contexts(self.contexts, self.json_path)
        print("Window closed – final save done")
        self.root.destroy()


if __name__ == "__main__":
    contexts = load_or_create_contexts()
    if contexts:
        Labeler(contexts)
    else:
        print("No contexts to label")
