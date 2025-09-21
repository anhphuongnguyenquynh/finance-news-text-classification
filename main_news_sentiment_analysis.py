from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Khởi tạo app
app = FastAPI()

# Load model & tokenizer 1 lần duy nhất
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Load fine-tuned weights
model_path = "sentiment_analysis_articles/model/bert_finetuned_sentiment.pth"
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)
model.eval()  # set model ở chế độ inference


class ArticleText(BaseModel):
    header: str
    text: str
    summary: str


@app.post("/article_sentiment")
def analysis(article: ArticleText):
    # Tokenize input
    inputs = tokenizer(article.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

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

# curl -X POST "http://127.0.0.1:8000/article_sentiment" \
#      -H "Content-Type: application/json" \
#      -d '{"text": "PBS to Cut 15% of Its Staff"}'