import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from collections import Counter
from settings import EMBEDDING_MODEL_NAME, JSON_PATH, CACHE_PATH, MODEL_SAVE_PATH, RESULTS_FILE

EMBEDDING_DIM = 384
HIDDEN_DIM = 256
NUM_LABELS = 2
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3

def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_input_texts(data):
    return [f"Number: {str(item['number']).strip()} | Context: {str(item['context']).strip()}" for item in data]

def compute_or_load_embeddings(texts, cache_path, model_name):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        cached_texts = loaded["texts"].tolist()
        if cached_texts == texts:
            print("Cache hit: embeddings match current data.")
            return loaded["embeddings"]
        print("Cache mismatch: recomputing embeddings.")
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    np.savez(cache_path, embeddings=embeddings, texts=np.array(texts, dtype=object))
    print(f"Saved embeddings to {cache_path}")
    return embeddings

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], idx

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_labels),
        )
    def forward(self, x):
        return self.network(x)

def compute_class_weights(labels, num_labels):
    counter = Counter(labels)
    total = len(labels)
    return torch.tensor([total / (num_labels * counter.get(i, 1)) for i in range(num_labels)], dtype=torch.float32)

def print_data_summary(data, labels):
    label_counts = Counter(labels)
    numbers_seen = Counter(str(item["number"]) for item in data)
    avg_ctx_len = np.mean([len(item["context"]) for item in data])
    print(f"Total samples: {len(data)}")
    print(f"Label 0: {label_counts.get(0,0)}")
    print(f"Label 1: {label_counts.get(1,0)}")
    print(f"Unique numbers: {len(numbers_seen)}")
    print(f"Most common numbers: {numbers_seen.most_common(5)}")
    print(f"Avg context length: {avg_ctx_len:.0f} chars")

def train_epoch(model, loader, optimizer, criterion, device, pretrain_preds):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for embeddings, labels, _ in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total

def pretrain_inference_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for embeddings, labels, _ in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            logits = model(embeddings)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total else 0.0

def evaluate_on_train(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_confidences = []
    with torch.no_grad():
        for embeddings, labels, _ in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            logits = model(embeddings)
            probs = torch.softmax(logits, dim=1)
            all_confidences.extend(probs.cpu().numpy())
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += len(labels)
    return total_loss/total, correct/total, all_confidences

def run_inference_and_save(model, embeddings_np, data, device, output_path):
    model.eval()
    all_embeddings = torch.tensor(embeddings_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(all_embeddings)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    results = [{"number": item["number"], "context": item["context"], "label": item["label"],
                "confidence_label_0": float(probs[i][0]),
                "confidence_label_1": float(probs[i][1]),
                "predicted_label": int(np.argmax(probs[i]))} for i, item in enumerate(data)]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    correct = sum(1 for r in results if r["predicted_label"]==r["label"])
    print(f"Re-evaluated train-set accuracy: {correct}/{len(results)} = {correct/len(results):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_json_data(JSON_PATH)
    labels = [int(item["label"]) for item in data]
    texts = build_input_texts(data)
    print_data_summary(data, labels)
    embeddings = compute_or_load_embeddings(texts, CACHE_PATH, EMBEDDING_MODEL_NAME)
    dataset = EmbeddingDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_weights = compute_class_weights(labels, NUM_LABELS).to(device)
    model = ClassifierHead(EMBEDDING_DIM, HIDDEN_DIM, NUM_LABELS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Pretrain Acc':>13} {'LR':>12}")
    print("-"*60)
    best_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        pretrain_acc = pretrain_inference_epoch(model, loader, device)
        train_loss, train_acc = train_epoch(model, loader, optimizer, criterion, device, pretrain_preds={})
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        marker = ""
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            marker = " *"
        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.4f} {pretrain_acc:>13.4f} {current_lr:>12.6f}{marker}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    _, _, all_confidences = evaluate_on_train(model, eval_loader, criterion, device)
    run_inference_and_save(model, embeddings, data, device, RESULTS_FILE)
    confidences_arr = np.array(all_confidences)
    print(f"Mean confidence label 0: {confidences_arr[:,0].mean():.4f}")
    print(f"Mean confidence label 1: {confidences_arr[:,1].mean():.4f}")

if __name__ == "__main__":
    main()

