from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
import pysolr

solr = pysolr.Solr('http://localhost:8983/solr/mycore', always_commit=True)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

client = MongoClient('mongodb+srv://ataemiruncu:<password>@cluster0.prqckre.mongodb.net/?retryWrites=true&w=majority')
db = client['Yazlab3']
users_collection = db['users']

interests = [
    "scheduling", "data mining", "computational complexity", "performance evaluation", 
    "approximation algorithms", "fault tolerance", "parallel algorithms", "model checking", 
    "distributed systems", "preconditioning", "routing", "optimization", "load balancing", 
    "semidefinite programming", "real-time systems", "machine learning", "parallel programming", 
    "logic programming", "clustering", "formal methods", "linear programming", "complexity", 
    "parallel processing", "mobile computing", "algorithms", "graph coloring", "scalability", 
    "parallel computing", "information retrieval", "iterative methods"
]


import fasttext
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

#Modeli Yükle
model = fasttext.load_model('cc.en.300.bin')

# SciBERT modelini yükle
model_name = "allenai/scibert_scivocab_uncased"
model2 = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

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

class Kullanici:
    def __init__(self, adsoyad, kullanici_adi, sifre, dogum_tarihi, ilgi_alanlari):
        self.adsoyad = adsoyad
        self.kullanici_adi = kullanici_adi
        self.sifre = sifre
        self.dogum_tarihi = dogum_tarihi
        self.ilgi_alanlari = ilgi_alanlari

    def __str__(self):
        return f"Kullanici(adsoyad={self.adsoyad}, kullanici_adi={self.kullanici_adi}, dogum_tarihi={self.dogum_tarihi}, ilgi_alanlari={self.ilgi_alanlari})"

# Precision hesaplama
def hesapla_precision(tahminler, gercek_etiketler):
    true_positives = sum([t and g for t, g in zip(tahminler, gercek_etiketler)])
    predicted_positives = sum(tahminler)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def benzerlik(areas):

    word_vectors = []
    for area in areas:
       embedding = model.get_word_vector(area)
       #print("İlgi alanının embedding'i:", embedding)
       word_vectors.append(embedding)

    interest_vector = np.mean(word_vectors, axis=0)

    # Her bir vektör için cosine similarity hesaplayın
    similarities = []
    for vector in vectors:
        similarity = cosine_similarity([interest_vector], [vector])[0][0]
        similarities.append(similarity)

    # En benzer 5 makaleyi bulun
    similarities_with_indices = [(similarity, index) for index, similarity in enumerate(similarities)]
    similarities_with_indices.sort(reverse=True)  # Benzerliklere göre sıralayın

    top_5_indices = [index for _, index in similarities_with_indices[:5]]  # En benzer 5 makalenin endekslerini alın
    top_5_Precision = [_ for _, index in similarities_with_indices[:5]]

    top_5_results = []

    for i in top_5_indices:
        top_5_results.append(articles[i])

    collection = db['article_vectors']
    top_5_article = []
    for id in top_5_results:
        query = {"_id": id}
        data = collection.find_one(query)
        top_5_article.append(data['article'])

    birlesik_liste = [f"{result} - {article} - Precision: {precision}" for result, article, precision in zip(top_5_results, top_5_article, top_5_Precision)]

    word_vectors2 = []
    for area in areas:
        tokens = tokenizer(area, return_tensors="pt")
        with torch.no_grad():
            output = model2(**tokens)
        embeddings = output.last_hidden_state.mean(dim=1).squeeze()
        word_vectors2.append(embeddings)
    
    interest_vector2 = sum(word_vectors2) / len(word_vectors2)

    # Vektörleri içeren dosyaların yolu
    vector_folder = "vectors"
    # Dosyaları gezerek cosine similarity hesapla
    similarities2 = []
    for filename in os.listdir(vector_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(vector_folder, filename), "r") as file:
                vector_values = [float(val) for val in file.readline().split()]
                vector = torch.Tensor(vector_values)
                if len(vector) > 0:
                    similarity = cosine_similarity(interest_vector2.reshape(1, -1), vector.reshape(1, -1))
                    similarities2.append((filename, similarity))

    similarities2.sort(key=lambda x: x[1], reverse=True)
    top_5_Precision2 = [index for _, index in similarities2[:5]]  # En benzer 5 makalenin endekslerini alın
    top_5_indices2 = [_ for _, index in similarities2[:5]]

    top_5_results2 = []
    for i in top_5_indices2:
        top_5_results2.append(i.split('.')[0])

    collection = db['article_vectors']
    top_5_article2 = []
    for id in top_5_results2:
        query = {"_id": id}
        data = collection.find_one(query)
        top_5_article2.append(data['article'])
    
    birlesik_liste2 = [f"{result} - {article} - Precision: {precision}" for result, article, precision in zip(top_5_results2, top_5_article2, top_5_Precision2)]

    

    return birlesik_liste, birlesik_liste2


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})
        if user and user['password']==password:
            session['username'] = username  # Oturumda kullanıcı adını sakla
            return redirect(url_for('profile', username=username))  # Profil sayfasına yönlendir
        else:
            flash('Yanlış giriş bilgileri. Lütfen tekrar deneyin.', 'error')
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        birth_date = request.form['birth_date']
        username = request.form['username']
        password = request.form['password']
        selected_interests = request.form.getlist('interests[]')  # Seçilen ilgi alanlarını al

        user_data = {
            'full_name': full_name,
            'birth_date': birth_date,
            'username': username,
            'password': password,
            'interests': selected_interests
        }

        users_collection.insert_one(user_data)
        flash('Kayıt başarıyla tamamlandı. Giriş yapabilirsiniz.', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', interests=interests)

@app.route('/profile/<username>')
def profile(username):
    if 'username' in session and session['username'] == username:
        return render_template('profile.html', username=username)
    else:
        return redirect(url_for('index'))  # Oturum açmamış kullanıcıyı ana sayfaya yönlendir
    
@app.route('/profile/<username>/profilyönetim', methods=['GET', 'POST'])
def profil_yönetim(username):
    if 'username' in session and session['username'] == username:
        username = session['username']
        user = users_collection.find_one({'username': username})
        if request.method == 'POST':
            selected_interests = request.form.getlist('interests[]')
            # Mevcut ilgi alanlarını al ve seçilenleri çıkar
            updated_interests = [interest for interest in user['interests'] if interest not in selected_interests]
            users_collection.update_one(
                {'username': username},
                {'$set': {'interests': updated_interests}}
            )
        return render_template('profilyönetim.html', user=user, interests=interests, username=username)
    else:
        return redirect(url_for('index'))  # Oturum açmamış kullanıcıyı ana sayfaya yönlendir


@app.route('/makaleonerisistemi', methods=['POST'])
def makale_öneri_sistemi():
    if 'username' in session:
        username = session['username']
        user_data = users_collection.find_one({'username': username})
        areas = user_data['interests']  # Kullanıcının ilgi alanlarını al

        # Benzerlik fonksiyonunu kullanarak ilgi alanlarına göre sonuçları hesapla
        birlesik_liste, birlesik_liste2 = benzerlik(areas)
        
        if request.form.getlist("feedback[]"):
            selected_articles = request.form.getlist("feedback[]")
            selected_articles_name = []
            for i in selected_articles:
                article = i.split("-")[0]
                selected_articles_name.append(article.strip())
                if i.split("-")[1] not in user_data['interests']:
                    user_data['interests'].append(i.split("-")[1])
                    users_collection.update_one({'username': username}, {"$set": {"interests": user_data["interests"]}})

            keys_folder = 'Krapivin2009/keys'
            article_keys = [article + '.key' for article in selected_articles_name]
            for article_key in article_keys:
                key_path = os.path.join(keys_folder, article_key)
                if os.path.exists(key_path):
                   with open(key_path, 'r') as file:
                        contents = file.read()
                        keywords = contents.split('\n')
                        for keyword in keywords:
                            if keyword != "" and keyword not in user_data['interests']:
                                user_data['interests'].append(keyword)
                                users_collection.update_one({'username': username}, {"$set": {"interests": user_data["interests"]}})
        

        return render_template('makale_oneri_sistemi.html', birlesik_liste=birlesik_liste, birlesik_liste2=birlesik_liste2)
    else:
        return redirect(url_for('index'))  # Oturum açmamış kullanıcıyı ana sayfaya yönlendir

@app.route('/makalearamasistemi', methods=['POST'])
def makale_arama_sistemi():
    if 'username' in session:
        username = session['username']
        return render_template('baslat.html', results=[])
    else:
        return redirect(url_for('index'))  # Oturum açmamış kullanıcıyı ana sayfaya yönlendir

@app.route('/search', methods=['POST'])
def search():
    if 'username' in session:
        username = session['username']
        query = request.form['query']
        interest = request.form.get('interest')
        params = {
        'defType': 'edismax',
        'q': f'{query}~0.8 OR {interest}~0.8',
        'qf': 'title keywords',
        'rows': 50
        }
        results = solr.search(**params)
        return render_template('makale_arama_sistemi.html', interests=interests, results=results)
    else:
        return redirect(url_for('index'))  # Oturum açmamış kullanıcıyı ana sayfaya yönlendir

@app.route('/result/<id>')
def result(id):
    if 'username' in session:
        username = session['username']
        user_data = users_collection.find_one({'username': username})
        doc = solr.search(f'id:{id}').docs[0]
        fname = id + '.key'
        keys_folder = 'Krapivin2009/keys'
        key_path = os.path.join(keys_folder, fname)
        if os.path.exists(key_path):
            with open(key_path, 'r') as file:
                contents = file.read()
                keywords = contents.split('\n')
                for keyword in keywords:
                    if keyword != "" and keyword not in user_data['interests']:
                        user_data['interests'].append(keyword)
                        users_collection.update_one({'username': username}, {"$set": {"interests": user_data["interests"]}})
        return render_template('result.html', doc=doc)
    else:
        return redirect(url_for('index'))  # Oturum açmamış kullanıcıyı ana sayfaya yönlendir

if __name__ == '__main__':
    app.run(debug=True)
