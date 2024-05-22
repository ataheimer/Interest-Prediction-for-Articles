import fasttext
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

#Modeli Yükle
model = fasttext.load_model('cc.en.300.bin')

# .txt dosyalarının olduğu klasör yolunu belirtin
data_folder = "NLP"

vectors = []
articles = []
# Tüm .txt dosyalarını işle
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        articles.append(filename.split('.')[0])
        with open(file_path, "r", encoding="utf-8") as file:
            # Dosyadan cümleleri oku ve her bir cümlenin embedding'ini çıkar
            for line in file:
                sentence_embedding = model.get_sentence_vector(line.strip())
                #print("Cümlenin embedding'i:", sentence_embedding)
                vectors.append(sentence_embedding)
                


interests = [
    "scheduling",
    "data mining",
    "computational complexity",
    "performance evaluation",
    "approximation algorithms",
    "fault tolerance",
    "parallel algorithms",
    "model checking",
    "distributed systems",
    "preconditioning",
    "routing",
    "optimization",
    "load balancing",
    "semidefinite programming",
    "real-time systems",
    "machine learning",
    "parallel programming",
    "logic programming",
    "clustering",
    "formal methods",
    "linear programming",
    "complexity",
    "parallel processing",
    "mobile computing",
    "algorithms",
    "graph coloring",
    "scalability",
    "parallel computing",
    "information retrieval",
    "iterative methods"
]

word_vectors = []
for area in interests:
    embedding = model.get_word_vector(area)
    #print("İlgi alanının embedding'i:", embedding)
    word_vectors.append(embedding)

average_vector = (word_vectors[0] + word_vectors[1] + word_vectors[8] + word_vectors[14] + word_vectors[21] + word_vectors[28] + word_vectors[29]) / 3

# Her bir vektör için cosine similarity hesaplayın
similarities = []
for vector in vectors:
    similarity = cosine_similarity([average_vector], [vector])[0][0]
    similarities.append(similarity)

# En benzer 5 makaleyi bulun
similarities_with_indices = [(similarity, index) for index, similarity in enumerate(similarities)]
similarities_with_indices.sort(reverse=True)  # Benzerliklere göre sıralayın

top_5_indices = [index for _, index in similarities_with_indices[:5]]  # En benzer 5 makalenin endekslerini alın
top_5_Precision = [_ for _, index in similarities_with_indices[:5]]

print("En benzer 5 makalenin endeksleri:", top_5_indices)
print(top_5_Precision)

top_5_results = []
for i in top_5_indices:
    top_5_results.append(articles[i])

print(top_5_results)

for index in top_5_indices:
    print("isimler:", articles[index])



