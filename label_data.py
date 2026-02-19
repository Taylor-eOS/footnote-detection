import json
import tkinter as tk
from tkinter import ttk
import re

def load_or_create_contexts(input_path="input.txt",json_path="number_contexts.json"):
    try:
        with open(input_path,"r",encoding="utf-8") as f:
            full_text=f.read()
    except Exception as e:
        print(f"Cannot read input.txt: {e}")
        return []
    tokens=full_text.split()
    number_occurrences=[]
    for i,token in enumerate(tokens):
        for match in re.finditer(r'-?\d+(?:[.,]\d+)*',token):
            number_occurrences.append((i,match.group(0)))
    contexts=[]
    for i,number in number_occurrences:
        start=max(0,i-10)
        end=min(len(tokens),i+11)
        before=tokens[start:i]
        after=tokens[i+1:end]
        text=" ".join(before+[number]+after)
        contexts.append({"input_text":text,"number":number,"label":0})
    return contexts

def save_contexts(contexts,json_path="number_contexts.json"):
    try:
        with open(json_path,"w",encoding="utf-8") as f:
            json.dump(contexts,f,ensure_ascii=False,indent=2)
        print(f"Saved {len(contexts)} entries")
    except Exception as e:
        print(f"Save failed: {e}")

class Labeler:
    def __init__(self,contexts,json_path="number_contexts.json"):
        self.contexts=contexts
        self.json_path=json_path
        self.page_size=8
        self.current_page=0
        self.total_pages=(len(contexts)+self.page_size-1)//self.page_size
        self.root=tk.Tk()
        self.root.title("Label 0/1 – click to toggle, SPACE = next")
        self.root.geometry("1200x750")
        self.labels=[ctx["label"] for ctx in contexts]
        self.row_widgets={}
        self.frame=ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        self.canvas=tk.Canvas(self.frame)
        self.scrollbar=ttk.Scrollbar(self.frame,orient="vertical",command=self.canvas.yview)
        self.scrollable_frame=ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>",lambda e:self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0,0),window=self.scrollable_frame,anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left",fill="both",expand=True)
        self.scrollbar.pack(side="right",fill="y")
        self.status=ttk.Label(self.root,text="",font=("TkDefaultFont",11))
        self.status.pack(pady=6)
        self.root.bind("<space>",self.next_page)
        self.root.protocol("WM_DELETE_WINDOW",self.on_closing)
        self.show_page()
        self.root.mainloop()

    def show_page(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.row_widgets={}
        start=self.current_page*self.page_size
        end=min(start+self.page_size,len(self.contexts))
        for idx in range(start,end):
            ctx=self.contexts[idx]
            tokens=ctx["input_text"].split()
            try:
                num_idx=tokens.index(ctx["number"])
            except ValueError:
                num_idx=len(tokens)//2
            before=" ".join(tokens[max(0,num_idx-10):num_idx])
            number=ctx["number"]
            after=" ".join(tokens[num_idx+1:num_idx+11])
            row_frame=ttk.Frame(self.scrollable_frame)
            row_frame.pack(fill=tk.X,pady=5,padx=5)
            var=tk.IntVar(value=self.labels[idx])
            chk=ttk.Checkbutton(row_frame,variable=var,command=lambda pos=idx,v=var:self.toggle_label(pos,v))
            chk.pack(side=tk.LEFT,padx=6)
            bg="#d4edda" if self.labels[idx]==1 else "#f8d7da"
            txt=tk.Text(row_frame,height=2,wrap=tk.WORD,font=("TkDefaultFont",13),bg=bg,relief=tk.FLAT,cursor="arrow",state=tk.NORMAL)
            txt.pack(side=tk.LEFT,fill=tk.X,expand=True)
            txt.tag_configure("context",foreground="#aaaaaa")
            txt.tag_configure("number",foreground="#c0392b",font=("TkDefaultFont",15,"bold"))
            txt.insert(tk.END,before+(" " if before else ""),"context")
            txt.insert(tk.END,number,"number")
            txt.insert(tk.END,(" " if after else "")+after,"context")
            txt.configure(state=tk.DISABLED)
            txt.bind("<Button-1>",lambda e,pos=idx,v=var,t=txt:self.toggle_via_click(pos,v,t))
            self.row_widgets[idx]=(var,txt)
        status=f"Page {self.current_page+1} / {self.total_pages}   Entries {start+1}–{end}"
        self.status.config(text=status)

    def toggle_label(self,pos,var):
        new_val=var.get()
        self.labels[pos]=new_val
        self.contexts[pos]["label"]=new_val
        if pos in self.row_widgets:
            _,txt=self.row_widgets[pos]
            txt.configure(bg="#d4edda" if new_val==1 else "#f8d7da")
        save_contexts(self.contexts,self.json_path)

    def toggle_via_click(self,pos,var,txt=None):
        new_val=1-var.get()
        var.set(new_val)
        self.toggle_label(pos,var)

    def next_page(self,event=None):
        if self.current_page<self.total_pages-1:
            self.current_page+=1
            self.show_page()
        else:
            save_contexts(self.contexts,self.json_path)
            self.status.config(text="Done – close window when ready")

    def on_closing(self):
        save_contexts(self.contexts,self.json_path)
        self.root.destroy()

if __name__=="__main__":
    contexts=load_or_create_contexts()
    if contexts:
        Labeler(contexts)
    else:
        print("No entries")
