"""API endpoints of the hiring challenge."""
import re
from io import BytesIO
from typing import Dict, List
import joblib
import pandas as pd
from fastapi.applications import FastAPI
from fastapi.param_functions import File
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

app = FastAPI()


def remove_tags(sentence: str) -> str:
    """Remove HTML tags from a sentence."""
    html_tag = '<.*?>'
    cleaned_sentence = re.sub(html_tag, ' ', sentence)
    return cleaned_sentence


def remove_punctuation(sentence: str) -> str:
    """Remove punctuation from a sentence."""
    cleaned_sentence = re.sub(r'[?|!\'"#]', '', sentence)
    cleaned_sentence = re.sub(r'[,|.;:(){}<>/]|-', ' ', cleaned_sentence)
    cleaned_sentence = cleaned_sentence.replace("\n", " ")
    return cleaned_sentence


def keep_alpha(sentence: str) -> str:
    """Keep only alphabetic characters in a sentence."""
    alpha_sentence = re.sub('[^a-z A-Z]+', ' ', sentence)
    return alpha_sentence


def lower_case(sentence: str) -> str:
    """Convert a sentence to lower case."""
    lower_case_sentence = sentence.lower()
    return lower_case_sentence


def lemmatize_words(sentence: str) -> str:
    """Lemmatize words in a sentence."""
    lem = WordNetLemmatizer()
    lemmatized_words = [lem.lemmatize(word, 'v') for word in sentence.split()]
    lemmatized_sentence = ' '.join(lemmatized_words)
    return lemmatized_sentence


def remove_stop_words(sentence: str) -> str:
    """Remove stop words from a sentence."""
    stop_words = set(stopwords.words('english'))
    stop_words.update(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                       'may', 'also', 'across', 'among', 'beside', 'however', 'yet', 'within'])
    no_stop_words = [word for word in sentence.split() if word not in stop_words]
    no_step_sentence = ' '.join(no_stop_words)
    return no_step_sentence


def text_preprocess(sentence: str) -> str:
    """Preprocess text by applying a series of cleaning functions."""
    pre_processed_sentence = remove_tags(sentence)
    pre_processed_sentence = remove_punctuation(pre_processed_sentence)
    pre_processed_sentence = keep_alpha(pre_processed_sentence)
    pre_processed_sentence = lower_case(pre_processed_sentence)
    pre_processed_sentence = lemmatize_words(pre_processed_sentence)
    pre_processed_sentence = remove_stop_words(pre_processed_sentence)

    return pre_processed_sentence


@app.post("/genres/train")
def train(file: bytes = File(...)) -> None:
    """Train a predictive model to rank movie genres based on their synopsis."""
    df = pd.read_csv(BytesIO(file))

    genres: List[List[str]] = []
    for i, genre in enumerate(df['genres']):
        genre = genre.split()
        genres.append(genre)
        df.at[i, 'genres'] = genre

    df['clean_synopsis'] = df['synopsis'].apply(text_preprocess)
    train_x: pd.Series = df['clean_synopsis']

    mlb = MultiLabelBinarizer()
    transformed = mlb.fit_transform(genres)

    for i, line in enumerate(transformed):
        df.at[i, 'genres'] = line

    df_new: DataFrame = df
    df_new[['Drama', 'Horror', 'Thriller', 'Children', 'Comedy', 'Romance', 'Action', 'Adventure', 'Crime', 'Film-noir',
            'Documentary', 'War', 'Fantasy', 'Animation', 'Mystery', 'Western', 'Sci-Fi', 'Musical']] = pd.DataFrame(
        df_new['genres'].tolist(), index=df_new.index)
    train_y = df_new.drop(['movie_id', 'year', 'synopsis', 'clean_synopsis', 'genres'], axis=1)

    pipeline: Pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2))),
        ('clf', OneVsRestClassifier(LogisticRegression(class_weight='balanced')))
    ])
    pipeline.fit(train_x, train_y)
    joblib.dump(pipeline, 'movie_genre_classifier.joblib')


@app.post("/genres/predict")
def predict(file: bytes = File(...), model_path: str = './movie_genre_classifier.joblib') -> Dict[int, Dict[int, str]]:
    """Predict genres for movies based on their synopsis."""
    model: Pipeline = joblib.load(model_path)
    df_pred = pd.read_csv(BytesIO(file))
    df_pred['clean_synopsis'] = df_pred['synopsis'].apply(text_preprocess)

    category_column = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                       'Western']

    prob = model.predict_proba(df_pred['clean_synopsis'])
    prob = pd.DataFrame(prob, columns=category_column)

    sorted_indices = prob.values.argsort(axis=1)[:, ::-1]
    predictions_dict: Dict[int, Dict[int, str]] = {}
    for i, movie_id in enumerate(df_pred['movie_id']):
        predictions_dict[movie_id] = {j: category_column[sorted_indices[i][j]] for j in range(5)}

    return predictions_dict
