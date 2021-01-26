#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:24:50 2018
@author: tittaya
"""

import tensorflow as tf
from tensorflow.python.client import device_lib
tf.enable_eager_execution()

import numpy as np
import pandas as pd
import matplotlib as plt

from pandas.io import gbq
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

import os
import re
import time
import MeCab
import json


print(tf.executing_eagerly())
print(tf.__version__)
print(tf.test.gpu_device_name())
# print(device_lib.list_local_devices())

current_dir = '/home/tittaya_mairittha/internship/qas'
# os.chdir(current_dir)

################ Download and prepare the dataset ################

# =============================================================================
# # Load data from google bigquery
# project_id = 'robot-personnel'
# client = bigquery.Client.from_service_account_json('robot-personnel-76469dafbe2b.json')
# 
# query = """ SELECT * FROM qa_data.free_answer WHERE gender == '男性' """
# 
# dataset = gbq.read_gbq(query, project_id)
# dataset.to_csv('female.csv', sep=',', encoding='utf-8-sig')
# =============================================================================

# read data
df = pd.read_csv(current_dir + '/female.csv')
df.columns

# Select column and rename
df = df[['theme_id', 'question', 'answer_val', 'age', 'gender', 'prefecture']]
df = df.rename(columns={'answer_val': 'answer'})

# Reference : https://gist.github.com/ryanmcgrath/982242
# UNICODE RANGE : DESCRIPTION 
# 3000-303F : punctuation
# 3040-309F : hiragana
# 30A0-30FF : katakana
# FF00-FFEF : Full-width roman + half-width katakana
# 4E00-9FAF : Common and uncommon kanji

def clean_question(text):
    text = re.split(r'[。]', text)
    text = clean_text(text[0]+'。')
    return text

def clean_text(text):
    unicode = u"([^\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\uFF00-\uFFEF\u4E00-\u9FAF])"
    text = re.sub(unicode, "", text)
    return text


# Cleaning the questions and answers
df['question'] = df['question'].apply(lambda x: clean_question(str(x)))
df['answer'] = df['answer'].apply(lambda x: clean_text(str(x)))

# Remove empty cell
filter = df["answer"] != ""
df = df[filter].reset_index(drop=True)
df.head()

# Filtering out the questions and answers that are too short or too long
MAX_LENGTH = 50
df = df[df['question'].map(len) < MAX_LENGTH]
df = df[df['answer'].map(len) < MAX_LENGTH]
df = df.reset_index(drop=True)

# print(df.shape)
num_examples = len(df.index)
# num_examples = 100

def preprocess_sentence(w):
    w = clean_text(w)
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

def tokenizer(sentence):
    
    sentence = sentence.split('<start>')[1]
    sentence = sentence.split('<end>')[0]
    
    temp = ''
    tagger = MeCab.Tagger('')
    tagger.parse('') 
    
    node = tagger.parseToNode(sentence)
    while node:
        word = str(node.surface)
        node = node.next
            
        temp += word + ' '
        
    return '<start> ' + temp + ' <end>'

# Return word pairs in the format: [QUESTION, ANSWER]
def create_dataset(num_examples):

    word_pairs = [['' for x in range(2)] for n in range(num_examples)] 

    for index, row in df.iterrows():
        word_pairs[index][0] = preprocess_sentence(row['question']) 
        word_pairs[index][1] = preprocess_sentence(row['answer']) 
        
        if index == num_examples - 1: break
    
    return word_pairs


# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa (e.g., 5 -> "dad") 
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(tokenizer(phrase).split(' '))
        
        self.vocab = sorted(self.vocab)
        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1
        
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)

def load_dataset(num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(num_examples)

    # index word using the class defined above    
    inp_lang = LanguageIndex(q for q, a in pairs)
    targ_lang = LanguageIndex(a for q, a in pairs)
    
    # Vectorize the input and target languages
    # Question 
    input_tensor = [[inp_lang.word2idx[q] if q in inp_lang.word2idx else inp_lang.word2idx['<unk>'] for q in tokenizer(q).split(' ') if q] for q, a in pairs]
    
    # Answer 
    target_tensor = [[targ_lang.word2idx[a] if a in targ_lang.word2idx else targ_lang.word2idx['<unk>'] for a in tokenizer(a).split(' ') if a] for q, a in pairs]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    
    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar


################ Limit the size of the dataset ################

# Load dataset
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(num_examples)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)

################ Create a tf.data dataset ################

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 256
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
# dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))

def gru(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, hidden_size)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

################ Define the optimizer and the loss function ################

optimizer = tf.train.AdamOptimizer()

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


################ Checkpoints ################
    
checkpoint_dir = current_dir + '/training_checkpoints/female'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

log_dir = current_dir + '/logs/female'
summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
summary_writer.set_as_default()
global_step = tf.train.get_or_create_global_step()

################ Training ################

EPOCHS = 10
RESTORE = True

def training(restore):
    if restore == False:
        for epoch in range(EPOCHS):
            start = time.time()
    
            hidden = encoder.initialize_hidden_state()
            total_loss = 0
    
            for (batch, (inp, targ)) in enumerate(dataset):
                loss = 0
    
                with tf.GradientTape() as tape:
                    enc_output, enc_hidden = encoder(inp, hidden)
    
                    dec_hidden = enc_hidden
    
                    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
    
                    # Teacher forcing - feeding the target as the next input
                    for t in range(1, targ.shape[1]):
                        # passing enc_output to the decoder
                        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
    
                        loss += loss_function(targ[:, t], predictions)
    
                        # using teacher forcing
                        dec_input = tf.expand_dims(targ[:, t], 1)
    
                batch_loss = (loss / int(targ.shape[1]))
    
                total_loss += batch_loss
    
                variables = encoder.variables + decoder.variables
    
                gradients = tape.gradient(loss, variables)
    
                optimizer.apply_gradients(zip(gradients, variables))
    
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
    
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
            with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                tf.contrib.summary.scalar('loss', total_loss / N_BATCH)
    else:
        # Restore the latest checkpoint and test
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

training(RESTORE)

################ Testing ################


def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    inputs = [inp_lang.word2idx[w] if w in inp_lang.word2idx else inp_lang.word2idx['<unk>'] for w in tokenizer(sentence).split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
        
        if predicted_id in targ_lang.idx2word:
            result += targ_lang.idx2word[predicted_id]
        else:
            predicted_id = targ_lang.word2idx['<unk>']

        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

# =============================================================================
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    fig.savefig('./images/fig.png')   # save the figure to file
    plt.close(fig) 

    # plt.show()
# =============================================================================


def generate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    
    sentence = preprocess_sentence(sentence)

    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        
    sentence = re.sub(r'<start>|<end>|<pad>', '', sentence)
    result = re.sub(r'<start>|<end>|<pad>', '', result)
    
    print('Input: {}'.format(sentence))
    print('Predicted answer: {}'.format(result))

    return result

def ask(question):
    
    result = generate(question, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    
    return result


def evaluate_randomly(num_examples=1):
    
    score = 0
    bleu = sentence_bleu
    smoothie = SmoothingFunction().method4

    dfr = df.sample(frac=1).reset_index(drop=True)

    for index, row in dfr.iterrows():
        if index == num_examples: break
        
        question = row['question']
        answer = row['answer']
        
        predicted = generate(question, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        predicted = re.sub(r'<start>|<end>|<pad>', '', predicted)

        try:
            score = bleu(answer, predicted, smoothing_function=smoothie)
        except ZeroDivisionError:
            score = 0

    return question, answer, predicted, score



def testing():

    scores = []
    bleu = sentence_bleu
    smoothie = SmoothingFunction().method4

    for (inp_row, targ_row) in zip(input_tensor_val, target_tensor_val):
        
        score = 0
        question = ''
        answer = ''
        
        for (q, a) in zip(inp_row, targ_row):
            question += inp_lang.idx2word[q]
            answer += targ_lang.idx2word[a]
        
        
        predicted = generate(question, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

        try:
            score = bleu(answer, predicted, smoothing_function=smoothie)
        except ZeroDivisionError:
            score = 0
    
        scores.append(score)

        print('question: {}'.format(question))
        print('answer: {}'.format(answer))
        print('predicted: {}'.format(predicted))
        print('score: {}'.format(score))
        print('\n')
    
    belu_score = np.mean(scores) * 100
    print(belu_score)








    