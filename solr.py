import pysolr
import os

# Solr sunucusunun URL'sini ve çekirdek (core) adını girin
solr_url = 'http://localhost:8983/solr/mycore'
solr = pysolr.Solr(solr_url, always_commit=True)


# Dosyaların bulunduğu dizin
docs_path = 'Krapivin2009/docsutf8'

# Dosyaları oku ve Solr'a ekle
for filename in os.listdir(docs_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(docs_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Her dosya için bir belge oluştur
            doc = {
                'id': filename.split('.')[0],  # Her dosya için benzersiz bir ID
                'content': content  # Dosya içeriği
            }
            # Solr'a belgeyi ekle
            solr.add([doc])
            print(f'{filename} dosyası Solr\'a yüklendi.')
