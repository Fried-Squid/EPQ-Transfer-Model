from tatoebatools import ParallelCorpus, tatoeba                                #for getting tatoeba data
from unidecode import unidecode as decode                                       #decoding tatoeba data
from re import sub                                                              #formatting tatoeba data
from random import shuffle                                                      #shuffling data
from collections import Counter                                                 #used to count tokens
import tensorflow as tf                                                         #model backend
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

from pipe import traverse                                                       #useful
import numpy as np                                                              #for messing with tensors
import gc                                                                       #keeps memory usage low
from tensorflow.keras import optimizers                                         #adam


import os, shutil
folders = ['spa_data/encoder_input_data','spa_data/decoder_input_data','spa_data/decoder_target_data']
def clear_dir(folder):
    for filename in os.listdir(folder):
        if ".gitignore" in filename:
            pass
        else:
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
for f in folders:
    clear_dir(f)


USE_FREQUENCY_RESTRICTION = False
latent_dim = 128
epochs = 3000
batch_size = 1024

tf.config.list_physical_devices('GPU')

global data
dataRaw = ParallelCorpus("eng","spa")

def process_sentence(s):
    s = decode(s.lower())
    s = sub(r'([!.?])', r' \1', s)
    s = sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = sub(r'\s+', r' ', s)
    s = s.strip()
    s = '<s>' +' '+ s +' '+'</s>'
    return s

data=[]
for s,t in dataRaw:                                                             #no length check here as actual number of pairs is very small
  data.append((process_sentence(s.text),process_sentence(t.text)))

def shuffleData():
  global data
  shuffle(data)
shuffleData();                                                                  #ensures that the data used in the first few epochs is always different
print("Example pair: %s" % str(data[0])[1:-1].replace("', ", "' --> '"))        #prints an example bitext pair

rawEn,rawSp = list(zip(*data));english,spanish = list(rawEn),list(rawSp)
english,spanish=list(map(lambda x:x.split(" "),english)),list(map(lambda x:x.split(" "),spanish))

#This codeblock removes sentences if their translations are too long.
zipped = sorted(list(zip(english,spanish)),key=lambda x:len(x[1]))
(english, spanish) = zip(*zipped[:round(len(zipped)*99/100)]) #strip longest hundreth
del zipped

#this codeblock replaces uncommon tokens and replaces them with <UNKNOWN>
flattened_english = list(english | traverse)
print(len(flattened_english))
flattened_spanish = list(spanish | traverse)
english_counter = Counter(flattened_english)
spanish_counter = Counter(flattened_spanish)
spa_words = list(spanish_counter.keys())
eng_words = list(english_counter.keys())

sorted_eng_words = sorted(eng_words, key=lambda x:english_counter[x], reverse=True)
sorted_spa_words = sorted(spa_words, key=lambda x:spanish_counter[x], reverse=True)
#calculates things
num_pairs=len(english)
#wee woo bullshit lines, changes the token of the 10th percentile common words to a 0
spanish_tokenizer=dict(zip(sorted_spa_words,list(list(range(1, round(len(spa_words)*9/10)+1)) + list(np.zeros((10000,),dtype=int)))))
english_tokenizer=dict(zip(sorted_eng_words,list(list(range(1, round(len(eng_words)*9/10)+1)) + list(np.zeros((10000,),dtype=int)))))

#normal code

max_english_sentence_length=max(list(map(len, english)))
max_spanish_sentence_length=max(list(map(len, spanish)))

num_encoder_tokens=len(set(english_tokenizer.values()))
num_decoder_tokens=len(set(spanish_tokenizer.values()))

#data stuff
segment_size = 500
segment_count = len(english)//segment_size

def onehot(seq):
  seq = list(map(lambda x:spanish_tokenizer[x], seq))
  out = np.zeros((max_spanish_sentence_length,num_decoder_tokens))
  i=0
  for token in seq:
    temp = np.zeros(num_decoder_tokens)
    temp[token]=1
    out[i]=temp;i+=1
  return out

for seg in range(segment_count):
    encoder_input_data = np.ndarray((segment_size,max_english_sentence_length))
    i=0
    for seq in english[segment_size*seg:segment_size*(seg+1)]:
      temp = list(map(lambda x:english_tokenizer[x], seq))
      zeros = [0]*(max_english_sentence_length - len(temp))
      encoder_input_data[i] = np.array(temp+zeros)
      i+=1
    np.save(f'spa_data/encoder_input_data/segment{str(seg)}.npy', encoder_input_data)
    if seg%10 == 0:
        print(f'encoder_input_data/segment{str(seg)}')
    del encoder_input_data
gc.collect()

for seg in range(segment_count):
    decoder_input_data = np.ndarray((segment_size,max_spanish_sentence_length))
    i=0
    for seq in spanish[segment_size*seg:segment_size*(seg+1)]:
      temp = list(map(lambda x:spanish_tokenizer[x], seq))
      zeros = [0]*(max_spanish_sentence_length - len(temp))
      decoder_input_data[i] = np.array(temp+zeros)
      i+=1
    np.save(f'spa_data/decoder_input_data/segment{str(seg)}.npy', decoder_input_data)
    if seg%10 == 0:
        print(f'decoder_input_data/segment{str(seg)}')
    del decoder_input_data
gc.collect()

for seg in range(segment_count):
    decoder_target_data = np.ndarray((segment_size,max_spanish_sentence_length,num_decoder_tokens),dtype=np.uint8)
    i=0
    for each in spanish[segment_size*seg:segment_size*(seg+1)]:
      seq=list(each[1:])
      decoder_target_data[i] = onehot(seq)
      i+=1
    np.save(f'spa_data/decoder_target_data/segment{str(seg)}.npy', decoder_target_data)
    if seg%10 == 0:
        print(f'decoder_target_data/segment{str(seg)}')
    del decoder_target_data

params_to_save = np.asarray([[segment_size, segment_count,None,None],[max_english_sentence_length, max_spanish_sentence_length, num_encoder_tokens, num_decoder_tokens]])

np.save('spa_data/params.npy',params_to_save)
