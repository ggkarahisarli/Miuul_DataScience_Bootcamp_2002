#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('datasets/Recommendation Systems Datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/Recommendation Systems Datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Adım 2: Modelleme
##############################

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)

accuracy.rmse(predictions)


svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################
# stocastistic gradient sürecinin iterasyon sayısı gibi bir sürü ön tanımlı parametre var
# n_epochs iterasyon sayısı
# lr_all: learning rate
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}


gs = GridSearchCV(SVD,
                  param_grid, # yukarıdaki parametreleri dene
                  measures=['rmse', 'mae'], #error hesapla
                  cv=3, # çapraz doğrulama 3 katlı
                  n_jobs=-1, #ful performans
                  joblib_verbose=True) #raporlama

gs.fit(data)

gs.best_score['rmse'] # en iyi hata
gs.best_params['rmse'] # bu hatayı veren parametreler
# Hiper parametreler optimize edildi

##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params['rmse']) # en iyi paremtre değerleri keyworded argüman ile giriliyor

data = data.build_full_trainset() #tüm veri setini train sete çevirdik artık test setine ihtiyacımız kalmadı
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)






