import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

from model_train import BiLSTM
from func_dataprep import preprocess_text, pad_input, map_tokens
from func_eval_predict import predict


# Load utils BiLSTM
with open("saved_model/utils_bilstm.pkl", "rb") as f:
    utils = pickle.load(f)

word2idx = utils["word2idx"]
id2label = utils["id2label"]
max_len = utils["max_len"]

# Load model architectures
model = BiLSTM(vocab_size=len(word2idx), embedding_dim = 300, hidden_dim=128, output_dim=len(id2label), num_layers=2, dropout_rate=0.5)

# Load model weights
model.load_state_dict(torch.load("saved_model/news_categorize_bilstm.pth", map_location=torch.device("cpu")))
model.eval()

# Test sample with prediction function

# sample_text = "IRS Launches Safety Review Amid Threats To Workers Linked To Conspiracy Theories"
# label, probs = predict(model, sample_text, word2idx, max_len=30, id2label=id2label, device="cpu")

# print("Predicted:", label)
# print("Probabilities:", probs)

# FAST API APP

app = FastAPI()
class ArticleText(BaseModel):
    header: str

@app.post("/news_category")
def predict_endpoint(input: ArticleText):
    _, label, probs = predict(model, input.header, word2idx, max_len, id2label)
    return {"label": label, "probabilities": probs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


