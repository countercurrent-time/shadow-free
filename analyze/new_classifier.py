import numpy as np
import torch
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


def compute_features(y, y_pred, y_preds_perturbed):
    """Compute features for MIA classification."""
    sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    v_y = extract_embeddings(y)
    v_y_pred = extract_embeddings(y_pred)
    v_y_preds_perturbed = [extract_embeddings(y_p) for y_p in y_preds_perturbed]

    similarities = [sim(v_y, v_y_p) for v_y_p in v_y_preds_perturbed]

    features = {
        "sim_y_ypred": sim(v_y, v_y_pred),
        "mean_sim": np.mean(similarities),
        "std_sim": np.std(similarities),
        "max_sim": np.max(similarities),
        "min_sim": np.min(similarities),
    }
    return np.array(list(features.values()))

# Example data (replace with real dataset)

# member_data = [("x1_member", "y1_member", ["y1_pert1", "y1_pert2"])]  # Members
# non_member_data = [("x1_nonmember", "y1_nonmember", ["y1_pert1", "y1_pert2"])]  # Non-members

import json

def load_txt(file_path):
    """Load data from a JSON lines file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

# Load data from JSON files
def load_json(file_path):
    """Load data from a JSON lines file."""
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

# Paths to the dataset files
input_dir = "../CodeCompletion-line/dataset/javaCorpus/0.01/20/"
true_file = input_dir + "train_CodeGPT-small-java_victim_infer.txt"
false_file = input_dir + "test_CodeGPT-small-java_victim_infer.txt"
true_gt_file = input_dir + "train_surrogate.json"
false_gt_file = input_dir + "test_surrogate.json"

# Load the data
true_data = load_txt(true_file)
false_data = load_txt(false_file)
true_gt_data = load_json(true_gt_file)
false_gt_data = load_json(false_gt_file)

group_size = 12

# Combine data into member_data and non_member_data
member_data = []
non_member_data = []

def pair_data(true_gt_file, true_data):
    paired_data = []
    grouped_lines = [true_data[i:i + group_size] for i in range(0, len(true_data), group_size)]
    
    for input, output in zip(true_gt_file, grouped_lines):
        if input['id'] == output['id']:
            paired_data.append((input['gt'], output[0], output[1:]))
        else:
            raise ValueError(f"Mismatched IDs: {input['id']} and {output['id']}")
    return paired_data
 

# Prepare member_data and non_member_data
member_data = pair_data(true_data, true_gt_data)
non_member_data = pair_data(false_data, false_gt_data)

print(f"Loaded {len(member_data)} member samples and {len(non_member_data)} non-member samples.")



data = []
labels = []

# Generate features for all samples
for (x, y, y_perturbed) in member_data:
    data.append(compute_features(x, y, y_perturbed))
    labels.append(1)

for (x, y, y_perturbed) in non_member_data:
    data.append(compute_features(x, y, y_perturbed))
    labels.append(0)

data = np.array(data)
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the MIA classifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:, 1]

tpr = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", tpr)
print("Classification Report:")
print(classification_report(y_test, y_pred))