from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

# SciBERT modelini yükle
model_name = "allenai/scibert_scivocab_uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Kelimeyi tokenize et ve vektör gömme değerini al
word = "scheduling"
tokens = tokenizer(word, return_tensors="pt")
with torch.no_grad():
    output = model(**tokens)
embeddings = output.last_hidden_state.mean(dim=1).squeeze()

# Vektörleri içeren dosyaların yolu
vector_folder = "vectors"

# Dosyaları gezerek cosine similarity hesapla
similarities = []
for filename in os.listdir(vector_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(vector_folder, filename), "r") as file:
            vector_values = [float(val) for val in file.readline().split()]
            vector = torch.Tensor(vector_values)
            # Vektörlerin boyutunu kontrol et
            if len(vector) > 0:
                similarity = cosine_similarity(embeddings.reshape(1, -1), vector.reshape(1, -1))[0][0]
                similarities.append((filename, similarity))


# Benzerlik derecelerine göre sırala ve en benzer 5 dosyanın ismini ve benzerlik oranlarını yazdır
similarities.sort(key=lambda x: x[1], reverse=True)
top_5_similar = similarities[:5]
print("En benzer 5 dosyanın ismi ve benzerlik oranları:")
for file, sim in top_5_similar:
    print(f"{file}: {sim}")

top_5_Precision2 = [index for _, index in similarities[:5]]
top_5_indices2 = [_ for _, index in similarities[:5]]

