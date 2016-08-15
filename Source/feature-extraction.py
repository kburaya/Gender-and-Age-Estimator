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
    # Calculate the most popular ngrams for age/sex classes
    # TODO uncomment if neccessary!
    # nltk.download()


    # dictionary of dictionaries for 1000 most popular ngrams
    age_ngrams_dict = dict()

    for i in range(1, 4):
        # Building dictionaries of ngrams
        if i not in age_ngrams_dict:
            age_ngrams_dict[i] = dict()
        else:
            continue

        for age in users_texts_by_age:
            age_ngram = get_ngrams(users_texts_by_age[age], i)
            age_ngram = list(reversed(sorted(age_ngram, key=age_ngram.get)))
            # left fisrt 1000 most popular ngrams
            if age_ngram.__len__() > 1000:
                age_ngram = age_ngram[0:1000]
            age_ngrams_dict[i][age] = [age_ngram]

    with open('Resources/' + 'age_ngrams_dict' + '.pkl', 'wb') as f:
        pickle.dump(age_ngrams_dict, f, pickle.HIGHEST_PROTOCOL)

    sex_ngrams_dict = dict()
    for i in range(1, 4):
        if i not in sex_ngrams_dict:
            sex_ngrams_dict[i] = dict()
        else:
            continue

        for sex in users_texts_by_sex:
            sex_ngrams = get_ngrams(users_texts_by_sex[sex], i)
            sex_ngrams = list(reversed(sorted(sex_ngrams, key=sex_ngrams.get)))
            if sex_ngrams.__len__() > 100:
                sex_ngrams = sex_ngrams[0:1000]
            sex_ngrams_dict[i][sex] = [sex_ngrams]

    with open('Resources/' + 'sex_ngrams_dict' + '.pkl', 'wb') as f:
        pickle.dump(sex_ngrams_dict, f, pickle.HIGHEST_PROTOCOL)
    return


def calculate_ngrams_features():
    file = open('Resources/age_ngrams_dict.pkl', 'rb')
    age_ngrams_dict = pickle.load(file)
    file = open('Resources/sex_ngrams_dict.pkl', 'rb')
    sex_ngrams_dict = pickle.load(file)

    #calculate average number of age/sex ngrams per message for every user
    global data, users_texts_by_age, users_texts_by_id, users_texts_by_sex
    data['avr_age_user_1grams'] = 0
    data['avr_age_user_2grams'] = 0
    data['avr_age_user_3grams'] = 0
    data['avr_age_user_1grams'] = data['avr_age_user_1grams'].astype(np.float)
    data['avr_age_user_2grams'] = data['avr_age_user_2grams'].astype(np.float)
    data['avr_age_user_3grams'] = data['avr_age_user_3grams'].astype(np.float)
    data['avr_sex_user_1grams'] = 0
    data['avr_sex_user_2grams'] = 0
    data['avr_sex_user_3grams'] = 0
    data['avr_sex_user_1grams'] = data['avr_sex_user_1grams'].astype(np.float)
    data['avr_sex_user_2grams'] = data['avr_sex_user_2grams'].astype(np.float)
    data['avr_sex_user_3grams'] = data['avr_sex_user_3grams'].astype(np.float)

    for index, row in data.iterrows():
        user = row['user']
        age = row['age']
        sex = row['sex']
        try:
            tweets = users_texts_by_id[user]
        except:
            continue

        #1grams
        num_user_age_1grams = 0
        num_user_sex_1grams = 0

        user_ngrams = get_ngrams(tweets, 1)

        for ngram in user_ngrams:
            if ngram in age_ngrams_dict[1][age][0]:
                num_user_age_1grams += 1
            if ngram in sex_ngrams_dict[1][sex][0]:
                num_user_sex_1grams += 1

        avr_user_age_1grams = (float)(num_user_age_1grams / tweets.__len__())
        data.avr_age_user_1grams[data.user == user] = avr_user_age_1grams
        avr_user_sex_1grams = (float)(num_user_sex_1grams / tweets.__len__())
        data.avr_sex_user_1grams[data.user == user] = avr_user_sex_1grams

        #2grams
        num_user_age_2grams = 0
        num_user_sex_2grams = 0
        user_ngrams = get_ngrams(tweets, 2)

        for ngram in user_ngrams:
            if ngram in age_ngrams_dict[2][age][0]:
                num_user_age_2grams += 1
            if ngram in sex_ngrams_dict[2][sex][0]:
                num_user_sex_2grams += 1

        avr_user_age_2grams = (float)(num_user_age_2grams / tweets.__len__())
        data.avr_age_user_2grams[data.user == user] = avr_user_age_2grams
        avr_user_sex_2grams = (float)(num_user_sex_2grams / tweets.__len__())
        data.avr_sex_user_2grams[data.user == user] = avr_user_sex_2grams

        #3grams
        num_user_age_3grams = 0
        num_user_sex_3grams = 0
        user_ngrams = get_ngrams(tweets, 3)

        for ngram in user_ngrams:
            if ngram in age_ngrams_dict[3][age][0]:
                num_user_age_3grams += 1
            if ngram in sex_ngrams_dict[3][sex]:
                num_user_sex_3grams += 1

        avr_user_age_3grams = (float)(num_user_age_3grams / tweets.__len__())
        data.avr_age_user_3grams[data.user == user] = avr_user_age_3grams
        avr_user_sex_3grams = (float)(num_user_sex_3grams / tweets.__len__())
        data.avr_sex_user_3grams[data.user == user] = avr_user_sex_3grams
        #AGE SECTION ENDS
        print (str(index) + " row succeed\n")

    data.to_csv('Resources/data_features_v1.csv', sep = '\t')
    return


def main():
    read_data()
    #calculate_mentions_feature()
    #calculate_ngrams_dicts()
    calculate_ngrams_features()
    return

if __name__ == "__main__":
    main()