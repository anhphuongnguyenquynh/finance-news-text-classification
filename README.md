# 📊 Finance News Text Classification

---
## 🚀 Features
📰 **Sentiment Analysis on Articles**  
  - Classify financial news & reports into **Positive / Neutral / Negative** sentiments using **BERT** fine-tuned on domain-specific data.  
  - Useful for traders, analysts, and portfolio managers to detect market mood.  

📰 **News category classfication on Articles headers and summary**
- Classifiy financial news into categories (Business & Finance, Politics, Travel,...)
- Dataset: [Huffpost](https://www.kaggle.com/datasets/rmisra/news-category-dataset/data) 210k news headlines from 2012 and 2022 from HuffPost. Each record in dataset consists of attributes: category, headline, authors, link, short_description, date.
- Experiments and results:

| Method | Accuracy | Precision | Recall |
| GRU | --- | --- | --- |
| LSTM | --- | --- | --- |
| BiLSTM | 0.6385 | 0.6232 | 0.6385 |
| Transformers | 0.6180 | 0.6085 | 0.6406 |

I also using FastAPI to get response prediction of news headlines. 

---

## 🛠️ Tech Stack
- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (REST API for predictions)  
- **Deep Learning Framework**: [PyTorch](https://pytorch.org/)  
- **Pretrained Models**: [HuggingFace Transformers (BERT)](https://huggingface.co/bert-base-uncased)  
- **Deployment**: Uvicorn / Docker  
- **Data**: Financial articles & news datasets (custom + open sources)  

