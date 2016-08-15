import pickle
import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
    stemmer = PorterStemmer()
    for text in texts:
        text = re.sub(r'@[A-Za-z0-9_-]*', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'#[A-Za-z0-9_-]*', '', text)
        text = re.sub(r'pic\S+', '', text)

        text = nltk.word_tokenize(text)
        for i in range(0, text.__len__()):
            text[i] = text[i].lower()
            text[i] = stemmer.stem(text[i])


        text = ' '.join([word for word in text if word not in cachedStopWords])
        text = ''.join(ch for ch in text if ch not in exclude)

        text = nltk.word_tokenize(text)

        ngrams_list = nltk.ngrams(text, n)
        ngrams_list = [ ''.join(grams) for grams in ngrams_list]
        for ngram in ngrams_list:
            if not ngram in ngrams:
                ngrams[ngram] = 1
            else:
                ngrams[ngram] += 1

    return ngrams


def calculate_ngrams_dicts():
    # NGRAMS FEATURE SECTION BEGIN
    # Calculate the most popular ngrams for age/sex classes
    # TODO uncomment if neccessary!
    # nltk.download()


    # dictionary of dictionaries for 1000 most popular ngrams
    age_1gram_dict = dict()
    age_2gram_dict = dict()
    age_3gram_dict = dict()

    for i in range(1, 3):
        # Building dictionaries of ngrams
        for age in users_texts_by_age:
            age_ngram = get_ngrams(users_texts_by_age[age], i)
            age_ngram = list(reversed(sorted(age_ngram, key=age_ngram.get)))
            # left fisrt 1000 most popular ngrams
            if age_ngram.__len__() > 1000:
                age_ngram = age_ngram[0:1000]
            if i == 1:
                age_1gram_dict[age] = [age_ngram]
            elif i == 2:
                age_2gram_dict[age] = [age_ngram]
            elif i == 3:
                age_3gram_dict[age] = [age_ngram]

    with open('Resources/' + 'age_1gram_dict' + '.pkl', 'wb') as f:
        pickle.dump(age_1gram_dict, f, pickle.HIGHEST_PROTOCOL)
    with open('Resources/' + 'age_2gram_dict' + '.pkl', 'wb') as f:
        pickle.dump(age_2gram_dict, f, pickle.HIGHEST_PROTOCOL)
    with open('Resources/' + 'age_3gram_dict' + '.pkl', 'wb') as f:
        pickle.dump(age_3gram_dict, f, pickle.HIGHEST_PROTOCOL)

        # NGRAMS FEATURE CALCULATION SECTION END

    #TODO calculate sex user ngrams
    return


def calculate_ngrams_features():
    file = open('Resources/age_1gram_dict.pkl', 'rb')
    age_1gram_dict = pickle.load(file)
    #calculate average number of age/sex ngrams per message for every user
    global data, users_texts_by_age, users_texts_by_id, users_texts_by_sex
    data['avr_age_user_1grams'] = 0
    data['avr_age_user_2grams'] = 0
    data['avr_age_user_3grams'] = 0
    data['avr_age_user_1grams'] = data['avr_age_user_1grams'].astype(np.float)
    data['avr_age_user_2grams'] = data['avr_age_user_2grams'].astype(np.float)
    data['avr_age_user_3grams'] = data['avr_age_user_3grams'].astype(np.float)

    for index, row in data.iterrows():
        user = row['user']
        age = row['age']
        sex = row['sex']
        try:
            tweets = users_texts_by_id[user]
        except:
            continue

        #1grams
        num_user_1grams = 0
        user_ngrams = get_ngrams(tweets, 1)
        age_ngrams_dict = age_1gram_dict[age]

        for ngram in user_ngrams:
            if ngram in age_ngrams_dict[0]:
                num_user_1grams += 1

        avr_user_1grams = (float)(num_user_1grams / tweets.__len__())
        data.avr_age_user_1grams[data.user == user] = avr_user_1grams
        print ("Success")

    return


def main():
    read_data()
    #calculate_mentions_feature()
    #calculate_ngrams_dicts()
    calculate_ngrams_features()
    return

if __name__ == "__main__":
    main()