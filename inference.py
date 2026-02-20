import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from train_model import ClassifierHead, build_input_texts, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_RATE, NUM_LABELS
from settings import EMBEDDING_MODEL_NAME, INPUT_JSON, MODEL_SAVE_PATH

def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_embeddings(texts, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    return torch.tensor(embeddings, dtype=torch.float32)

def run_inference(model, embeddings, data, device):
    model.eval()
    embeddings = embeddings.to(device)
    with torch.no_grad():
        logits = model(embeddings)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    results = []
    correct = 0
    for i, item in enumerate(data):
        pred_label = int(np.argmax(probs[i]))
        true_label = int(item.get("label", -1))
        if true_label >= 0 and pred_label == true_label:
            correct += 1
        results.append({
            "number": item.get("number"),
            "context": item.get("context"),
            "label": true_label,
            "confidence_label_0": float(probs[i][0]),
            "confidence_label_1": float(probs[i][1]),
            "predicted_label": pred_label
        })
    if correct > 0:
        accuracy = correct / len(data)
        print(f"Accuracy on input data: {correct}/{len(data)} = {accuracy:.4f}")
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_json_data(INPUT_JSON)
    texts = build_input_texts(data)
    embeddings = compute_embeddings(texts, EMBEDDING_MODEL_NAME)
    model = ClassifierHead(EMBEDDING_DIM, HIDDEN_DIM, NUM_LABELS, DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    results = run_inference(model, embeddings, data, device)
    for r in results:
        conf = max(r["confidence_label_0"], r["confidence_label_1"])
        match = "✓" if r["predicted_label"] == r["label"] else "✗"
        print(f"{match} number={r['number']}  true={r['label']}  pred={r['predicted_label']}  conf={conf:.4f}")

if __name__ == "__main__":
    main()
