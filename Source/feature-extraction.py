import pickle
import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#GLOBAL VARIABLES SECTION
users_texts_by_age = dict()
users_texts_by_sex = dict()
users_texts_by_id = dict()
data = pd.DataFrame()
#GLOBAL VARIABLES SECTION END

def extract_mentions (text):
    try:
        mentions = re.findall('@[A-Za-z0-9_-]*', text)
        return mentions.__len__()
    except:
        return 0

def read_data():
    file = open('Resources/users_texts_by_age.pkl', 'rb')
    global users_texts_by_age
    users_texts_by_age = pickle.load(file)
    file = open('Resources/users_texts_by_sex.pkl', 'rb')
    global users_texts_by_sex
    users_texts_by_sex = pickle.load(file)
    file = open('Resources/users_texts_by_id.pkl', 'rb')
    global users_texts_by_id
    users_texts_by_id = pickle.load(file)
    global data
    data = pd.read_csv('Resources/data.csv')
    data = pd.DataFrame(data)


def calculate_mentions_feature():
    # MENTIONS FEATURES SECTION BEGIN
    # Calculate average number of mentions per user/age group/sex group
    global data, users_texts_by_age, users_texts_by_id, users_texts_by_sex
    data['avr_user_mentions'] = 0
    data['avr_user_mentions'] = data['avr_user_mentions'].astype(np.float)
    for index, row in data.iterrows():
        user = row['user']
        mentions = 0
        tweets_size = 0
        try:
            tweets = users_texts_by_id[user]
        except:
            continue
        for tweet in tweets:
            mentions += extract_mentions(tweet)
            tweets_size += 1
        avr_mentions = (float)(mentions / tweets_size)
        data.avr_user_mentions[data.user == user] = avr_mentions

    data['avr_age_group_mentions'] = 0
    data['avr_age_group_mentions'] = data['avr_age_group_mentions'].astype(np.float)
    for age in users_texts_by_age:
        age_summary_mentions = 0
        age_summary_tweets = 0
        for tweet in users_texts_by_age[age]:
            age_summary_tweets += 1
            age_summary_mentions += extract_mentions(tweet)

        age_avr_mentions = (float)(age_summary_mentions / age_summary_tweets)
        data.avr_age_group_mentions[data.age == age] = age_avr_mentions

    data['avr_sex_group_mentions'] = 0
    data['avr_sex_group_mentions'] = data['avr_sex_group_mentions'].astype(np.float)
    for sex in users_texts_by_sex:
        sex_summary_mentions = 0
        sex_summary_tweets = 0
        for tweet in users_texts_by_sex[sex]:
            sex_summary_tweets += 1
            sex_summary_mentions += extract_mentions(tweet)

        sex_avr_mentions = (float)(sex_summary_mentions / sex_summary_tweets)
        data.avr_sex_group_mentions[data.sex == sex] = sex_avr_mentions

        # MEANTIONS FEATURE SECTION END
    return


def get_ngrams(texts, n):
    ngrams = dict()
    cachedStopWords = stopwords.words("english")
    exclude = set(string.punctuation)
    wordnet_lemmatizer = WordNetLemmatizer()
    for text in texts:
        text = re.sub('@[A-Za-z0-9_-]*', '', text)
        text = re.sub(r'http\S+', '', text)
        text = nltk.word_tokenize(text)
        text = ' '.join([word for word in text if word not in cachedStopWords])
        text = ''.join(ch for ch in text if ch not in exclude)


        ngrams_list = nltk.ngrams(text, n)
        ngrams_list = [ ' '.join(grams) for grams in ngrams_list]
        for ngram in ngrams_list:
            if not ngram in ngrams:
                ngrams[ngram] = 1
            else:
                ngrams[ngram] += 1

    return ngrams

def main():
    read_data()
    #calculate_mentions_feature()

    #NGRAMS FEATURE SECTION BEGIN
    #Calculate the most popular ngrams for age/sex classes
    #TODO uncomment if neccessary!
    #nltk.download()

    age_ngram_dict = dict()
    for i in range(1, 2):
        #Building dictionaries of ngrams
        for age in users_texts_by_age:
            age_ngram = get_ngrams(users_texts_by_age[age], i)
            age_ngram = list(reversed(sorted(age_ngram, key = age_ngram.get)))
            print ("Success")


    print ("Success")
    return

if __name__ == "__main__":
    main()