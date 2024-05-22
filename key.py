import glob
from collections import Counter

# Anahtar kelimeleri depolamak için bir Counter nesnesi oluşturalım
keyword_counter = Counter()

# keys klasöründeki .key uzantılı dosyaları alalım
key_files = glob.glob('Krapivin2009/keys/*.key')

# Her dosyayı açıp içindeki anahtar kelimeleri bulup Counter'a ekleyelim
for file_path in key_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # İçerikten anahtar kelimeleri ayıklayalım ("enter" tuşuna göre ayıracağız)
        keywords = content.split('\n')  # "enter" tuşuna göre ayır
        keyword_counter.update(keywords)

# En çok tekrar eden 10 anahtar kelimeyi alalım ve frekanslarını yazdıralım
top_keywords = keyword_counter.most_common(31)
for keyword, frequency in top_keywords:
    print((f"Anahtar Kelime: {keyword} | Frekans: {frequency}"))
