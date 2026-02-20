import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from collections import Counter

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
JSON_PATH = "ground_truth.json"
CACHE_PATH = "embeddings_cache.npz"
MODEL_SAVE_PATH = "classifier_head.pt"
EMBEDDING_DIM = 384
HIDDEN_DIM = 256
NUM_LABELS = 2
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3

def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_input_texts(data):
    texts = []
    for item in data:
        number = str(item["number"]).strip()
        context = str(item["context"]).strip()
        combined = f"Number: {number} | Context: {context}"
        texts.append(combined)
    return texts

def compute_or_load_embeddings(texts, cache_path, model_name):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        cached_texts = loaded["texts"].tolist()
        if cached_texts == texts:
            print("Cache hit: embeddings match current data.")
            return loaded["embeddings"]
        else:
            print("Cache mismatch: recomputing embeddings.")
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(texts)} samples...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False,)
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
        return self.embeddings[idx], self.labels[idx]

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim // 2, num_labels),)

    def forward(self, x):
        return self.network(x)

def compute_class_weights(labels, num_labels):
    counter = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_labels):
        count = counter.get(i, 1)
        weights.append(total / (num_labels * count))
    return torch.tensor(weights, dtype=torch.float32)

def print_data_summary(data, labels):
    label_counts = Counter(labels)
    numbers_seen = Counter(str(item["number"]) for item in data)
    print(f"Total samples:        {len(data)}")
    print(f"Label 0:   {label_counts.get(0, 0)}")
    print(f"Label 1:  {label_counts.get(1, 0)}")
    print(f"Unique numbers:       {len(numbers_seen)}")
    print(f"Most common numbers:  {numbers_seen.most_common(8)}")
    avg_ctx_len = np.mean([len(item["context"]) for item in data])
    print(f"Avg context length:   {avg_ctx_len:.1f} chars")

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total

def evaluate_on_train(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_confidences = []
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            logits = model(embeddings)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_confidences.extend(probs.cpu().numpy().tolist())
            total_loss += loss.item() * len(labels)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total, all_confidences

def run_inference_and_save(model, embeddings_np, data, device, output_path):
    model.eval()
    all_embeddings = torch.tensor(embeddings_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(all_embeddings)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    results = []
    for i, item in enumerate(data):
        result = {
            "number": item["number"],
            "context": item["context"],
            "label": item["label"],
            "confidence_label_0": float(probs[i][0]),
            "confidence_label_1": float(probs[i][1]),
            "predicted_label": int(np.argmax(probs[i])),
        }
        results.append(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved inference results to {output_path}")
    correct = sum(1 for r in results if r["predicted_label"] == r["label"])
    print(f"Train-set accuracy (re-evaluated): {correct}/{len(results)} = {correct/len(results):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data = load_json_data(JSON_PATH)
    labels = [int(item["label"]) for item in data]
    texts = build_input_texts(data)
    print_data_summary(data, labels)
    embeddings = compute_or_load_embeddings(texts, CACHE_PATH, EMBEDDING_MODEL_NAME)
    print(f"Embedding matrix shape: {embeddings.shape}")
    dataset = EmbeddingDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    class_weights = compute_class_weights(labels, NUM_LABELS).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    model = ClassifierHead(EMBEDDING_DIM, HIDDEN_DIM, NUM_LABELS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("\nStarting training...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'LR':>12}")
    print("-" * 44)
    best_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        marker = " *" if train_loss < best_loss else ""
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.4f} {current_lr:>12.6f}{marker}")
    print(f"\nBest training loss: {best_loss:.4f}")
    print(f"Best model saved to {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    _, _, all_confidences = evaluate_on_train(model, eval_loader, criterion, device)
    run_inference_and_save(model, embeddings, data, device, "inference_results.json")
    confidences_arr = np.array(all_confidences)
    mean_conf_0 = confidences_arr[:, 0].mean()
    mean_conf_1 = confidences_arr[:, 1].mean()
    print(f"\nMean confidence for label 0: {mean_conf_0:.4f}")
    print(f"Mean confidence for label 1: {mean_conf_1:.4f}")

main()
