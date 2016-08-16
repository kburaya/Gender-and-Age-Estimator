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

cachedStopWords = stopwords.words("english")
exclude = set(string.punctuation)
stemmer = PorterStemmer()
#GLOBAL VARIABLES SECTION END

def prepare_text(text):
    global cachedStopWords, exclude, stemmer
    text = re.sub(r'@[A-Za-z0-9_-]*', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#[A-Za-z0-9_-]*', '', text)
    text = re.sub(r'pic\S+', '', text)

    text = nltk.word_tokenize(text)
    for i in range(0, text.__len__()):
        text[i] = text[i].lower()
        text[i] = stemmer.stem(text[i])

    return text


def get_mentions (text):
    try:
        mentions = re.findall('@[A-Za-z0-9_-]*', text)
        return mentions.__len__()
    except:
        return 0


def read_data():
    global users_texts_by_age, users_texts_by_sex, users_texts_by_id, data
    file = open('Resources/users_texts_by_age.pkl', 'rb')
    users_texts_by_age = pickle.load(file)
    file = open('Resources/users_texts_by_sex.pkl', 'rb')
    users_texts_by_sex = pickle.load(file)
    file = open('Resources/users_texts_by_id.pkl', 'rb')
    users_texts_by_id = pickle.load(file)
    data = pd.read_csv('Resources/data.csv')
    data = pd.DataFrame(data)


def define_features():
    # Define feature columns in dataset
    global data
    data['user_1grams'] = 0
    data['user_2grams'] = 0
    data['user_3grams'] = 0
    data['avr_mentions'] = 0
    data['avr_punctuation'] = 0


def calculate_mentions_feature():
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
            mentions += get_mentions(tweet)
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
            age_summary_mentions += get_mentions(tweet)

        age_avr_mentions = (float)(age_summary_mentions / age_summary_tweets)
        data.avr_age_group_mentions[data.age == age] = age_avr_mentions

    data['avr_sex_group_mentions'] = 0
    data['avr_sex_group_mentions'] = data['avr_sex_group_mentions'].astype(np.float)
    for sex in users_texts_by_sex:
        sex_summary_mentions = 0
        sex_summary_tweets = 0
        for tweet in users_texts_by_sex[sex]:
            sex_summary_tweets += 1
            sex_summary_mentions += get_mentions(tweet)

        sex_avr_mentions = (float)(sex_summary_mentions / sex_summary_tweets)
        data.avr_sex_group_mentions[data.sex == sex] = sex_avr_mentions

        # MEANTIONS FEATURE SECTION END
    return


def get_ngrams(texts, n):
    # Calculates dictionary of ngrams for list of texts
    ngrams = dict()
    cachedStopWords = stopwords.words("english")
    exclude = set(string.punctuation)
    stemmer = PorterStemmer()
    for text in texts:
        text = prepare_text(text)
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


def get_punctuation(text):
    # Calculate number of punctuations in given text
    punctuation_num = 0
    text = prepare_text(text)
    punctuation = ''.join(ch for ch in text if ch  in exclude)
    punctuation_num += punctuation.__len__()
    return punctuation_num


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


def calculate_common_ngrams_dict():
    global users_texts_by_id
    common_ngrams_dict = dict()
    count = 0
    for user in users_texts_by_id:
        count += 1
        texts = users_texts_by_id[user]
        for i in range(1, 4):
            if i not in common_ngrams_dict:
                common_ngrams_dict[i] = dict()
            user_ngrams = get_ngrams(texts, i)
            for ngram in user_ngrams:
                if ngram not in common_ngrams_dict[i]:
                    common_ngrams_dict[i][ngram] = 1
                else:
                    common_ngrams_dict[i][ngram] += 1
        print (str(count) + ' ' + str(user) + ' ngrams calculated')

    for i in range(1, 4):
        common_ngrams_dict[i] = list(reversed(sorted(common_ngrams_dict[i], key=common_ngrams_dict[i].get)))
        if common_ngrams_dict[i].__len__() > 1000:
            common_ngrams_dict[i] = common_ngrams_dict[i][0:1000]

    with open('Resources/' + 'common_ngrams_dict' + '.pkl', 'wb') as f:
        pickle.dump(common_ngrams_dict, f, pickle.HIGHEST_PROTOCOL)
    return


def calculate_features():
    # Calculates features for every user in dataset
    global data, users_texts_by_id
    file = open('Resources/common_ngrams_dict.pkl', 'rb')
    common_ngrams_dict = pickle.load(file)
    count = 0
    for index, row in data.iterrows():
        count += 1
        user = row['user']
        try:
            texts = users_texts_by_id[user]
        except:
            continue

        # ngrams feature section
        num_ngrams = 0
        for i in range(1, 4):
            user_ngrams = get_ngrams(texts, i)
            for ngram in user_ngrams:
                if ngram in common_ngrams_dict[i]:
                    num_ngrams += 1

            avr_ngrams = float(num_ngrams / len(texts))

            if i == 1:
                data.user_1grams[data.user == user] = avr_ngrams
            elif i == 2:
                data.user_2grams[data.user == user] = avr_ngrams
            elif i == 3:
                data.user_3grams[data.user == user] = avr_ngrams
        # ngram feature section end

        # punctuation/mentions feature section
        num_punctuations = 0
        num_mentions = 0
        for text in texts:
            num_punctuations += get_punctuation(text)
            num_mentions += get_mentions(text)
        avr_punctuation = float(num_punctuations / len(texts))
        avr_mentions = float(num_mentions / len(texts))
        data.avr_punctuation[data.user == user] = avr_punctuation
        data.avr_mentions[data.user == user] = avr_mentions
        # punctuation/mentions feature section end

        print(str(count) + ' ' + str(user) + ' user calculated')

    data.to_csv('Resources/data_features.csv', sep = '\t')
    return


def main():
    read_data()
    define_features()

    # TODO uncomment to get dictionary files
    #calculate_common_ngrams_dict()
    # calculate_ngrams_dicts()

    calculate_features()
    #calculate_ngrams_features()
    return

if __name__ == "__main__":
    main()