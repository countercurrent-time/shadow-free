import numpy as np
import torch
import json
from transformers import AutoModel, AutoTokenizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

def compute_tfidf_similarity(x, y):
    # Use TfidfVectorizer to compute TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([x, y])

    # Convert TF-IDF matrix to array for easier interpretation
    tfidf_array = tfidf_matrix.toarray()

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Display results
    # print("TF-IDF Matrix:")
    # print(tfidf_array)

    # print("\nFeature Names:")
    # print(vectorizer.get_feature_names_out())

    # print("\nCosine Similarity between the two strings:")
    # print(cosine_sim[0][0])
    return cosine_sim[0][0]


# Example data (replace with real dataset)
# input_file = '/kaggle/working/shadow-free/Classifier/classifier_save/javaCorpus/CodeGPT-small-java/20/res_20_50.json'
# output_file = '/kaggle/working/shadow-free/Classifier/classifier_save/javaCorpus/CodeGPT-small-java/20/similarity.json'

# with open(input_file) as f:
#     data = f.readlines()

# data = [d for d in data if '"label": 0, "predicition_label": 0' in d]
# new_data = []

# for d in data:
#     d = json.loads(d)
#     # d['similarity'] = float(compute_similarity(d['gt'], d['prediction']))
#     d['similarity'] = float(compute_tfidf_similarity(d['gt'], d['prediction']))
#     new_data.append(d)

# new_data = sorted(new_data, key=lambda x: x['similarity'])

# with open(output_file, 'w') as f:
#     for d in new_data:
#         f.write(json.dumps(d))
#         f.write('\n')



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

def pair_data(true_data, true_gt_data):
    paired_data = []
    grouped_lines = [true_data[i:i + group_size] for i in range(0, len(true_data), group_size)]
    
    if len(true_gt_data) != len(grouped_lines):
        raise ValueError(f"Mismatched lengths: f{len(true_gt_data)} and f{len(grouped_lines)}")

    for input, output in zip(true_gt_data, grouped_lines):
        paired_data.append((input['gt'], output[0], output[1:]))
    return paired_data

# Prepare member_data and non_member_data
member_data = pair_data(true_data, true_gt_data)
non_member_data = pair_data(false_data, false_gt_data)

print(f"Loaded {len(member_data)} member samples and {len(non_member_data)} non-member samples.")

true_data = []
false_data = []
# Generate features for all samples
for (x, y, y_perturbed) in member_data:
    true_data.append(compute_similarity(x, y))

for (x, y, y_perturbed) in non_member_data:
    false_data.append(compute_similarity(x, y))

# true_data = sorted(true_data)
# false_data = sorted(false_data)

with open('similarity_true.txt') as f:
    for i in true_data:
        f.write(i)

with open('similarity_false.txt') as f:
    for i in false_data:
        f.write(i)
