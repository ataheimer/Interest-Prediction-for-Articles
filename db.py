from pymongo import MongoClient

client = MongoClient('mongodb+srv://ataemiruncu:18temmuz2003@cluster0.prqckre.mongodb.net/?retryWrites=true&w=majority')
db = client['Yazlab3']
collection = db['article_vectors']

import os
import codecs

folder_path = 'Krapivin2009/docsutf8'  # Dosya yolunu buraya girin

for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        with codecs.open(file_path, 'r', 'utf-8') as file:
            lines = file.readlines()
            if len(lines) > 1:  # Dosyanın en az iki satırı varsa
                _id = file_name.split('.')[0]  # Dosya adından _id oluştur
                article = lines[1].strip()  # Dosyanın ikinci satırındaki metni al
                # MongoDB'ye kaydet
                collection.insert_one({'_id': _id, 'article': article})
