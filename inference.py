import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from train_model import ClassifierHead, build_input_texts, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_RATE, NUM_LABELS
from settings import EMBEDDING_MODEL_NAME, INPUT_JSON, MODEL_SAVE_PATH, OUTPUT_JSON

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
    for i, item in enumerate(data):
        results.append({
            "number": item.get("number"),
            "context": item.get("context"),
            "confidence_label_0": float(probs[i][0]),
            "confidence_label_1": float(probs[i][1]),
            "predicted_label": int(np.argmax(probs[i]))
        })
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_json_data(INPUT_JSON)
    texts = build_input_texts(data)
    embeddings = compute_embeddings(texts, EMBEDDING_MODEL_NAME)
    model = ClassifierHead(EMBEDDING_DIM, HIDDEN_DIM, NUM_LABELS, DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    results = run_inference(model, embeddings, data, device)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Inference complete. Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
