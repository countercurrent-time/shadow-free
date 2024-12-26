import numpy as np
import torch
import json
from transformers import AutoModel, AutoTokenizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# Initialize the pre-trained embedding model (e.g., CodeBERT)
MODEL_NAME = "microsoft/codebert-base"
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 将模型移动到 GPU

def extract_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)  # 使用 GPU 计算
    return outputs.squeeze(0).cpu().numpy()  # 转回 CPU 再转换为 NumPy

def compute_similarity(x, y):
    sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    v_x = extract_embeddings(x)
    v_y = extract_embeddings(y)
    return sim(v_x, v_y)


# Example data (replace with real dataset)
input_file = '/kaggle/working/shadow-free/Classifier/classifier_save/javaCorpus/CodeGPT-small-java/20/res_20_50.json'
output_file = '/kaggle/working/shadow-free/Classifier/classifier_save/javaCorpus/CodeGPT-small-java/20/similarity.json'

with open(input_file) as f:
    data = f.readlines()

data = [d for d in data if '"label": 0, "predicition_label": 0' in d]
new_data = []

for d in data:
    d = json.loads(d)
    d['similarity'] = compute_similarity(d['gt'], d['prediction'])
    new_data.append(d)

new_data = sorted(new_data, key=lambda x: x['similarity'])

with open(output_file, 'w') as f:
    for d in new_data:
        f.write(json.dumps(d))
        f.write('\n')
