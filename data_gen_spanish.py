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
folders = ['~/env/Transfer Model/data/encoder_input_data','~/env/Transfer Model/data/decoder_input_data','~/env/Transfer Model/data/decoder_target_data']
def clear_dir(folder)
    for filename in os.listdir(folder):
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
english,spanish=list(map(lambda x:x.split(" "),english)),list(map(lambda x:x.split(" "),spanish)) #ace what the fuck

flattened_english = list(english | traverse)
flattened_spanish = list(spanish | traverse)
num_pairs=len(english)
english_counter=Counter(flattened_english)
spanish_counter=Counter(flattened_spanish)

spa_words = list(spanish_counter.keys())
eng_words = list(english_counter.keys())

spanish_tokenizer=dict(zip(sorted(spa_words),list(range(0, len(spa_words)))))
english_tokenizer=dict(zip(sorted(eng_words),list(range(0, len(eng_words)))))

max_english_sentence_length=max(list(map(len, english)))
max_spanish_sentence_length=max(list(map(len, spanish)))

print(max_english_sentence_length, max_spanish_sentence_length )


num_encoder_tokens=len(english_tokenizer)
num_decoder_tokens=len(spa_words)

#data stuff
segment_size = 500
segment_count = len(english)//500
print(segment_count*segment_size)

for i in range(segment_count):
    encoder_input_data = np.ndarray((segment_size,max_english_sentence_length))
    gc.collect()
    i=0
    for seq in english:
      temp = list(map(lambda x:english_tokenizer[x], seq))
      zeros = [0]*(max_english_sentence_length - len(temp))
      encoder_input_data[i] = np.array(temp+zeros)
      i+=1
    np.save(f'~/env/Transfer Model/data/encoder_input_data/segment{str(i)}.npy', encoder_input_data)

decoder_input_data = np.ndarray((maxPairs,max_spanish_sentence_length))
gc.collect()

i=0
for seq in spanish[:maxPairs]:
  temp = list(map(lambda x:spanish_tokenizer[x], seq))
  zeros = [0]*(max_spanish_sentence_length - len(temp))
  decoder_input_data[i] = np.array(temp+zeros)
  i+=1
def onehot(seq):
  seq = list(map(lambda x:spanish_tokenizer[x], seq))
  out = np.zeros((max_spanish_sentence_length,num_decoder_tokens))
  i=0
  for token in seq:
    temp = np.zeros(num_decoder_tokens)
    temp[token]=1.0
    out[i]=temp;i+=1
  return out
decoder_target_data = np.ndarray((maxPairs,max_spanish_sentence_length,num_decoder_tokens))
gc.collect()

i=0
for each in spanish[:maxPairs]:
  seq=list(each[1:])
  decoder_target_data[i] = onehot(seq)
  i+=1

#This is an offset check, the second line (target data) should be offset by a single timestep.
print(str(list(map(lambda x:f'{int(x):03}', decoder_input_data[5]))).replace("'",""))
def f(x):
  try: return f'{int(x.index(1.0)):03}'
  except: return '000'
print(str(list(map(lambda x:f(list(x)),list(decoder_target_data[5])))).replace("'",""))
