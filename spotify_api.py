import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
import requests
import neattext.functions as nfx
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from langdetect import detect


cid ='8d566e2e160442969e4c4ad9eba0131d'
secret ='2c1a8a77b6d54d0cbecd0e78782259e1'

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

def get_album_tracks(uri_info):
    uri = []
    track = []
    duration = []
    explicit = []
    track_number = []
    one = sp.playlist_tracks(uri_info)
    df1 = pd.DataFrame(one)

    for track in df1["items"]:
        track_uri = track["track"]["uri"]
        track_name = track["track"]["name"]
        artist_uri = track["track"]["artists"][0]["uri"]
        artist_info = sp.artist(artist_uri)

        artist_name = track["track"]["artists"][0]["name"]
        artist_pop = artist_info["popularity"]
        artist_genres = artist_info["genres"]

        album = track["track"]["album"]["name"]

        track_pop = track["track"]["popularity"]

    return df1

def scrape_lyrics(artistname, songname):
    artistname2 = str(artistname.replace(' ','-')) if ' ' in artistname else str(artistname)
    songname2 = str(songname.replace(' ','-')) if ' ' in songname else str(songname)
    songname2 = songname2.lower()
    page = requests.get('https://genius.com/'+ artistname2 + '-' + songname2 + '-' + 'lyrics')
    html = BeautifulSoup(page.text, 'html.parser')
    lyrics1 = html.find("div", class_="lyrics")
    lyrics2 = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-5 Dzxov")
    if lyrics1:
        lyrics = lyrics1.get_text()
    elif lyrics2:
        lyrics = lyrics2.get_text(separator='\n')
    elif lyrics1 == lyrics2 == None:
        lyrics = None
    return lyrics

def lyrics_onto_frame(df1):
    for i,x in enumerate(df1["items"]):
        track = x["track"]["name"]
        artist = x["track"]["artists"][0]["name"]
        test = scrape_lyrics(artist, track)
        df1.loc[i, 'lyrics'] = test     
    return df1

df1_tracks = get_album_tracks('spotify:playlist:5NZzXehKFHsdQPQfeyRFtU') 
df_lyrics = lyrics_onto_frame(df1_tracks)
df_l = df_lyrics.lyrics

df = pd.read_csv("../PLaiLIST/emotion-dataset.csv")
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
X = df['Clean_Text']
y = df['Emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10000, test_size = 0.2, shuffle=False)
pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression(solver='lbfgs', max_iter=10000))])
pipe_lr.fit(X_train, y_train)

emotions = dict()
for i in df_l:
    if i is None:
        print("None")
    elif (isinstance(i, float) or detect(i) != "en"):
        print('Not Valid')
    else:
        emotion = pipe_lr.predict([i])[0]
        if emotion in emotions:
            emotions[emotion] += 1
        else:
            emotions.update({emotion: 1})
for i,x in emotions.items():
    print(i, ":", x)
