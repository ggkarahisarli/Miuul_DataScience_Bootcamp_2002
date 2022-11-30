#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################



###############################################################
# İş Problemi
###############################################################

# Yeni kurulmuş bir online film izleme platformu kullanıcılarına
# film önerilerinde bulunmak istemektedir.
# Kullanıcıların login oranı çok düşük olduğu için kullanıcı alışkanlıklarını
# toplanemamaktadır. Bu sebeble iş birlikçi filtreleme yöntemleri ile ürün
# önerileri geliştirememektedir. Bireylerin beğenme alışkanlıkları yok.
# Fakat kullanıcıların tarayıcıdaki izlerinden (cookie id) hangi filmleri izledikleri
# biliniyor. Bu bilgiye göre film önerilerinde bulunulacaktır.

# Veri Seti Hikayesi
# 45000 film ile ilgili temel bilgileri barındırıyor.
# https://www.kaggle.com/rounakbanik/the-movies-dataset

###############################################################
# Değişkenler
###############################################################

# overview: film açıklamalarını içeriyor.


#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması


#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None) #Bütun sütunları yana doğru göster
pd.set_option('display.width', 500) #yanyana 500 tane gösterebilirsin
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("datasets/Recommendation Systems Datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape

df["overview"].head()
# TfidfVectorize methodu ile dilde yaygın kullanılan kelimeleri yoksayabiliriz.
# Ölçüm değeri olmayan sütun sayısını arttıran kelimeleri yok sayacağız.

tfidf = TfidfVectorizer(stop_words="english")

# df[df['overview'].isnull()]
#Overview ı boş olanları boşlukla doldurduk.
df['overview'] = df['overview'].fillna('')
#tfidf matrixini oluşturuyor.
tfidf_matrix = tfidf.fit_transform(df['overview'])

tfidf_matrix.shape

df['title'].shape

tfidf.get_feature_names()

# Matrisi elde ettik.
tfidf_matrix.toarray()


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

cosine_sim.shape
cosine_sim[1]


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

indices = pd.Series(df.index, index=df['title'])

indices.index.value_counts()
# Filmlerden fazla sayıda var.
#tittlelarda çoklama var. Bu çoklamalardan kurtulmak istiyoruz.
#En son çekilen filmi alıyoruz.
indices = indices[~indices.index.duplicated(keep='last')]

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]
#Sherlock ile diğer filmler arasındaki similarity scorelarına eriştik.
cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
