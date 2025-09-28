import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import Optional

from model_train import BiLSTM
from func_dataprep import preprocess_text, pad_input, map_tokens
from func_eval_predict import predict

#### LOAD MODEL AND FUNCTION FOR NEWS CATEGORY PREDICTION
# Load utils BiLSTM
with open("saved_model/utils_bilstm.pkl", "rb") as f:
    utils = pickle.load(f)

word2idx = utils["word2idx"]
id2label = utils["id2label"]
max_len = utils["max_len"]

# Load model architectures
cate_model = BiLSTM(vocab_size=len(word2idx), embedding_dim = 300, hidden_dim=128, output_dim=len(id2label), num_layers=2, dropout_rate=0.5)

# Load model weights
cate_model.load_state_dict(torch.load("saved_model/news_categorize_bilstm.pth", map_location=torch.device("cpu")))
cate_model.eval()

#Test sample with prediction function

# sample_text = "IRS Launches Safety Review Amid Threats To Workers Linked To Conspiracy Theories"
# label, probs = predict(model, sample_text, word2idx, max_len=30, id2label=id2label, device="cpu")

# print("Predicted:", label)
# print("Probabilities:", probs)

##### NEWS SENTIMENT ANALYSIS
# Load model & tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Load fine-tuned weights
model_path = "sentiment_analysis_articles/model/bert_finetuned_sentiment.pth"
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
sentiment_model.load_state_dict(state_dict, strict=False)
sentiment_model.eval()  # set model inference mode

# FAST API APP

app = FastAPI()
class ArticleText(BaseModel):
    header: Optional[str] = None
    text: Optional[str] = None
    summary: Optional[str] = None


# class ArticleText(BaseModel):
#     header: str

@app.post("/news_category")
def predict_endpoint(input: ArticleText):
    label, probs = predict(cate_model, input.header, word2idx, max_len, id2label)
    return {"label": label, "probabilities": probs}

@app.post("/article_sentiment")
def analysis(article: ArticleText):
    # Tokenize input
    inputs = tokenizer(article.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    # Convert logits -> label
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=1).item()

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    return {
        "text": article.text,
        "predicted_class": id2label[predicted_class],
        "probabilities": probs.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# curl -X POST "http://127.0.0.1:8000/news_category" \
#      -H "Content-Type: application/json" \
#      -d '{"header": "IRS Launches Safety Review Amid Threats To Workers Linked To Conspiracy Theories"}'

# curl -X POST "http://127.0.0.1:8000/article_sentiment" \
#      -H "Content-Type: application/json" \
#      -d '{"text": "PBS to Cut 15% of Its Staff"}'