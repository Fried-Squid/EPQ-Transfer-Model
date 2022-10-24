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

maxPairs = 15000
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

encoder_input_data = np.ndarray((maxPairs,max_english_sentence_length))
gc.collect()

i=0
for seq in english[:maxPairs]:
  temp = list(map(lambda x:english_tokenizer[x], seq))
  zeros = [0]*(max_english_sentence_length - len(temp))
  encoder_input_data[i] = np.array(temp+zeros)
  i+=1
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

#actual model fitting
gc.collect()
from tensorflow.keras import optimizers

#encoder embedding and input layers
encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, latent_dim,input_length=max_english_sentence_length)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,return_state=True)(x)
encoder_states = [state_h, state_c]

#decoder embedding and dense layer (STOP SETTING THE DENSE NEURON COUNT TO ONE ACE)
decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, latent_dim,input_length=max_spanish_sentence_length)(decoder_inputs)
x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

#compile the model and optimizer
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=["accuracy"])

#just summary things
model.summary()
from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="~/env/Transfer Model/checkpoints/",
    verbose=1,
    save_weights_only=False,
    save_freq='epoch',period=500)


model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[cp_callback]
          )
