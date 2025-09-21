import torch
import torch.nn.functional as F
import json
from model import BiLSTM

def predict(model, text, word2idx, max_len, idx2label, device="cpu"):

    # Preprocess: tokenize simple split (replace with nltk or your preprocessing pipeline)
    tokens = text.lower().split()
    token_ids = [word2idx.get(w, 0) for w in tokens]

    # Pad or truncate
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]

    # Convert to tensor
    input_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)  # shape [1, seq_len]

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)  # logits
        probs = F.softmax(outputs, dim=1).squeeze(0)  # shape [num_classes]
        pred_idx = torch.argmax(probs).item()
    
    predicted_label = idx2label[pred_idx]
    return predicted_label, probs.tolist()

###Load dictionary
with open("news_category_classification/word2idx.json", "r") as f:
    word2idx = json.load(f)

with open("news_category_classification/id2label.json", "r") as f:
    id2label = json.load(f)

model_categorizer = BiLSTM(vocab_size=len(word2idx), embedding_dim = 300, hidden_dim=128, output_dim = 42, num_layers=2, dropout_rate=0.5)
model_categorizer_path = "news_category_classification/model/news_categorize_bilstm.pth"
#load state dict
state_dict = torch.load(model_categorizer_path, map_location=torch.device("cpu"))
model_categorizer.load_state_dict(state_dict, strict=False)
model_categorizer.eval()my



if __name__ == "__main__":
    sample_text = "The World Wants More Vaccines. An Anti-Vaccine America Isn’t Helping."
    label, probs = predict(model_categorizer, sample_text, word2idx, max_len=30, idx2label=id2label, device="cpu")

    print("Predicted:", label)
    print("Probabilities:", probs)
