import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# İngilizce stopwords'leri yükle
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# İngilizce kök bulma için stemmer oluştur
stemmer = PorterStemmer()

# Dosya yolları
input_path = 'Krapivin2009/docsutf8/'
output_path = 'NLP/'

# Eğer NLP klasörü yoksa oluştur
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Dosyaları oku ve işle
for filename in os.listdir(input_path):
    if filename.endswith('.txt'):
        with open(os.path.join(input_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            
            # Rakamları kaldırma
            text = re.sub(r'\d+', '', text)

            # Stopwords ayıklama
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            filtered_text = ' '.join(filtered_words)
            
            # Noktalama işaretleri ayıklama
            filtered_text = re.sub(r'[^\w\s]', '', filtered_text)
            
            # Kök bulma
            stemmed_words = [stemmer.stem(word) for word in filtered_text.split()]
            stemmed_text = ' '.join(stemmed_words)
            
            # İşlenmiş metni NLP klasörüne yazdır
            output_filename = os.path.join(output_path, filename)
            with open(output_filename, 'w', encoding='utf-8') as output_file:
                output_file.write(stemmed_text)
