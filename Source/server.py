import pika
import nltk
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import os

# GLOBAL VARIABLES SECTION
cachedStopWords = stopwords.words("english")
exclude = set(string.punctuation)
stemmer = PorterStemmer()
common_ngrams_dict = dict()


# GLOBAL VARIABLES SECTION ENDS


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


def get_mentions(text):
    try:
        mentions = re.findall('@[A-Za-z0-9_-]*', text)
        return mentions.__len__()
    except:
        return 0


def get_ngrams_number(text, n):
    # Calculate number of common ngrams in message from common_ngrams_dict
    text = prepare_text(text)

    text = ' '.join([word for word in text if word not in cachedStopWords])
    text = ''.join(ch for ch in text if ch not in exclude)

    text = nltk.word_tokenize(text)

    ngrams_list = nltk.ngrams(text, n)
    ngrams_list = [''.join(grams) for grams in ngrams_list]

    global common_ngrams_dict
    common_ngrams = common_ngrams_dict[n]
    common_ngrams_num = 0
    for ngram in ngrams_list:
        if ngram in common_ngrams:
            common_ngrams_num += 1

    return common_ngrams_num


def get_punctuation(text):
    # Calculate number of punctuations in given text
    punctuation_num = 0
    text = prepare_text(text)
    punctuation = ''.join(ch for ch in text if ch in exclude)
    punctuation_num += punctuation.__len__()
    return punctuation_num


def get_ngrams(texts, n):
    # Calculates dictionary of ngrams for list of texts
    ngrams = dict()
    global cachedStopWords, exclude, stemmer
    for text in texts:
        text = prepare_text(text)
        text = ' '.join([word for word in text if word not in cachedStopWords])
        text = ''.join(ch for ch in text if ch not in exclude)

        text = nltk.word_tokenize(text)

        ngrams_list = nltk.ngrams(text, n)
        ngrams_list = [''.join(grams) for grams in ngrams_list]
        for ngram in ngrams_list:
            if not ngram in ngrams:
                ngrams[ngram] = 1
            else:
                ngrams[ngram] += 1
    return ngrams


def calculate_pos_tag_features(text):
    # calculate average number of each part of speech in user message, using https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    pos_tag_dict = dict()
    global exclude
    # looks life prepare text function but without stemming
    text = re.sub(r'@[A-Za-z0-9_-]*', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#[A-Za-z0-9_-]*', '', text)
    text = re.sub(r'pic\S+', '', text)
    text = nltk.word_tokenize(text)

    text = nltk.pos_tag(text)
    for pos_tag in text:
        if pos_tag[1] in exclude:
            continue
        if pos_tag[1] not in pos_tag_dict:
            pos_tag_dict[pos_tag[1]] = 1
        else:
            pos_tag_dict[pos_tag[1]] += 1

    return pos_tag_dict


def calculate_features(message):
    # Calculate features for model
    features = []
    file = open('Resources/common_ngrams_dict.pkl', 'rb')
    global common_ngrams_dict
    common_ngrams_dict = pickle.load(file)

    # Calculate ngrams feature
    for i in range(1, 4):
        features.append(get_ngrams_number(message, i))
    # Calculate ngram feature end

    # Calculate mentions feature
    features.append(get_mentions(message))

    # Calculate punctuation
    features.append(get_punctuation(message))

    # Calculate text size
    features.append(len(message))

    # Calculate starts with capital
    if len(message) > 0:
        if message[0].isupper():
            features.append(1)
        else:
            features.append(0)
    else:
        features.append(0)

    # Calculate ends with punctuation
    global exclude
    if len(message) > 0:
        if message[len(message) - 1] in exclude:
            features.append(1)
        else:
            features.append(0)
    else:
        features.append(0)

    # Calculate average capitals
    features.append(0)

    # Calculate words count
    features.append(len(message.split(' ')))

    # calculate vocabulary richness
    vocabulary = get_ngrams(message, 1)
    vocabulary_size = len(vocabulary)
    unique_words = sum(x == 1 for x in vocabulary.values())
    vocabulary_richness = float(unique_words / vocabulary_size) * 100
    features.append(vocabulary_richness)

    # Calculate pos-tag feature
    pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
                'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
                'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', ',', '.', ')', '(', ':',
                '$']

    pos_tag_dict = calculate_pos_tag_features(message)
    for pos_tag in pos_tags:
        if pos_tag in pos_tag_dict:
            features.append(pos_tag_dict[pos_tag])
        else:
            features.append(0)

    return features


def callback(ch, method, properties, body):
    print(" [x] Received %r" % (body,))
    answer = dict()
    message = str(body)
    features = calculate_features(message[2:-2])
    features = np.array(features).reshape(1, -1)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    file = 'Resources/grid_age_svm.pkl'
    age_model = joblib.load(file)

    file = 'Resources/log_model_sex.pkl'
    sex_model = joblib.load(file)

    answer['age'] = [age_model.predict(features)]
    age = answer['age'][0][0:1][0]

    age = str(age).replace("\r", "")
    age = str(age).replace("\n", "")
    answer['gender'] = [sex_model.predict(features)]
    gender = answer['gender'][0][0:1][0]

    print('Age: ' + str(age) + ', gender: ' + str(gender))
    print(" [x] Done")
    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    os.chdir('../');
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='messages_queue', durable=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(callback,
                          queue='messages_queue')

    channel.start_consuming()


if __name__ == "__main__":
    main()
