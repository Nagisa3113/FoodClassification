import os
import numpy as np
import tensorflow as tf
import googletrans
from nltk.corpus import stopwords
from tensorflow.keras import layers, losses, optimizers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM, Bidirectional
from sklearn.model_selection import StratifiedKFold, train_test_split

# trans = googletrans.Translator()

food = ('水煮肉片',
        '糖醋排骨',
        '排骨莲藕汤',
        '黄金虾仁炒饭',
        '锅包肉',
        '薯条',
        '藤椒鱼',
        '麻辣小龙虾',
        '宫保鸡丁',
        '番茄紫菜蛋汤',
        )

os_dir = '/Users'
# os_dir='/home'
base_dir = os_dir + '/nagisa/PycharmProjects/FC'

maxlen = 10
training_samples = 2400
validation_samples = 100
max_words = 2000
embedding_dim = 50


def stop(texts, lang='english'):
    stopset = set(stopwords.words(lang))
    for i, sentences in enumerate(texts):
        a = ''.join(sentences)
        new_sent = [word for word in a.split() if word not in stopset]
        texts[i] = ' '.join(new_sent)
    return texts


def to_one_hot(labels, dimension=10):
    for i, label in enumerate(labels):
        result = np.zeros(dimension)
        result[label] = 1
        labels[i] = result
    return labels


def load_data(data_dir, repeat=1):
    texts = []
    labels = []
    for i in range(repeat):
        for i in range(10):
            f = open(os.path.join(data_dir, str(i) + '.txt'), encoding='utf-8')
            lines = f.readlines()
            for line in lines:
                texts.append(line)
                labels.append(i)
            f.close()
    return texts, labels


train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
texts, labels = load_data(train_dir, repeat=4)

texts = stop(texts)
labels = to_one_hot(labels)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

glove_dir = os.path.join(base_dir, 'glove')
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)
print(embedding_matrix)

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
# db_train = tf.data.Dataset.from_tensor_slices((data, labels)).batch(16)

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# model.add(SimpleRNN(32, dropout=0.1, return_sequences=True))
# model.add(SimpleRNN(32, dropout=0.1))
# model.add(Dense(64))
# model.add(LSTM(32, return_sequences=True))
model.add(LSTM(48, dropout=0.1))
# model.add(Dense(32))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              # loss='binary_crossentropy',
              loss='categorical_crossentropy',
              metrics=['acc'])

# kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=5)
# for train, test in kfold.split(x_train, y_train):
#     history = model.fit(x_train[train], y_train[train],
#                         epochs=1,
#                         # batch_size=4,
#                         # validation_split=0.2,
#                         validation_data=(x_train[test], y_train[test]))

model.load_weights('pre.h5')


def process(msg):
    user_input = []
    user_input.append(msg)
    user_input = stop(user_input)
    user_out = pad_sequences(tokenizer.texts_to_sequences(user_input), maxlen=maxlen)
    result = model(user_out)
    # print(result)
    foodname = food[np.argmax(result)]
    prob = np.round(result[0][np.argmax(result)], 2)
    print(msg)
    print(foodname)
    print(prob)
    return foodname, prob


def input_test():
    user_input = []
    x = input('input:  ')
    # gx = trans.translate(x)
    # print(gx)
    # user_input.append(gx.text)
    user_input.append(x)
    user_input = stop(user_input)
    user_out = pad_sequences(tokenizer.texts_to_sequences(user_input), maxlen=maxlen)
    result = model(user_out)
    print(result)
    print(food[np.argmax(result)])
    print(np.round(result[0][np.argmax(result)], 2))


if __name__ == '__main__':
    # history = model.fit(data, labels,
    #                     epochs=10,
    #                     batch_size=64,
    #                     validation_split=0.2
    #                     # validation_data=(x_test, y_test)
    #                     )
    #
    # model.save_weights('pre.h5')
    #
    # # test_dir = os.path.join(base_dir, 'test')
    # labels = []
    # texts = []
    # texts, labels = load_data(test_dir)
    # texts = stop(texts)
    # labels = to_one_hot(labels)
    #
    # sequences = tokenizer.texts_to_sequences(texts)
    # x_test = pad_sequences(sequences, maxlen=maxlen)
    # y_test = np.asarray(labels)
    #
    # print(model.evaluate(x_test, y_test))

    while True:
        input_test()
