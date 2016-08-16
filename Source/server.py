import pika
import nltk
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
import numpy as np


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


def get_mentions (text):
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
    punctuation = ''.join(ch for ch in text if ch  in exclude)
    punctuation_num += punctuation.__len__()
    return punctuation_num


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
    return features


def callback(ch, method, properties, body):
    print(" [x] Received %r" % (body,))
    message = str(body)
    features = calculate_features(message[2:-2])
    print (features)
    features = np.array(features).reshape(1, -1)
    # TODO insert models here
    print(" [x] Done")
    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
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
