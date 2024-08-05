from os.path import join, dirname, abspath
import pickle
import pandas as pd

current_dir = dirname(abspath(__file__))

with open(join(current_dir, 'Models/Movies_recommendation_objects.pkl'), 'rb') as file:
    linear_sim, indices, df = pickle.load(file)

data_genres = df.iloc[:, 3:].columns
data_names = df['title'].tolist()


def get_recommendations(input):
    for name in data_names:
        if name == input:
            rec = get_recommendations_from_name(input)
            return rec
    
    input= input.replace(" ", "")
    words = input.split(",")
    unique_words = list(dict.fromkeys(words))
    valid_genres = [genre for genre in unique_words if genre in data_genres]
    if valid_genres != []:
        rec = get_recommendations_from_genres(valid_genres)
        return rec
    return pd.DataFrame()

def get_recommendations_from_name(title, linear_sim=linear_sim):
    idx = indices[title]
    sim_scores = list(enumerate(linear_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    df['genres'] = df[data_genres].apply(lambda row: '-'.join(row.index[row == 1]), axis=1)
    return df[['title', 'genres', 'rating']].iloc[movie_indices]

def get_recommendations_from_genres(genres):
    df_filtered = df.copy()
    for genre in genres:
        df_filtered = df_filtered[df_filtered[genre] == 1]
    df_filtered['genres'] = df_filtered[genres].apply(lambda row: '-'.join(row.index[row == 1]), axis=1)
    # df_recommendations = df_filtered[df_filtered['rating'] >= threshold_rating]
    df_recommendations = df_filtered.sort_values(by='rating', ascending=False)
    return df_recommendations[['title', 'genres', 'rating']].head(10)


# while True:
#     i = input("test: ")
#     rec = get_recommendations(i)
#     print(rec)
