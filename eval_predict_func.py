import torch
import torch.nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from dataprep_func import preprocess_text

def predict(model, text, word2idx, max_len, id2label, device="cpu"):
    # Tokenize
    tokens = preprocess_text(text)

    # Map to ids
    unk_id = word2idx.get("<UNK>", 0)
    token_ids = [word2idx.get(w, unk_id) for w in tokens]

    # Pad / truncate
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]

    # To tensor (Long for embedding)
    input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)  # shape [1, num_classes]
        probs = F.softmax(outputs, dim=1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
    
    predicted_label = id2label.get(pred_idx, "UNKNOWN")
    return predicted_label, probs.tolist()