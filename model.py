
import tensorflow as tf                                                         #model backend
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np                                                              #for messing with tensors
import gc                                                                       #keeps memory usage low
from tensorflow.keras import optimizers                                         #adam
import os

def train(epochs, batch_size, latent_dim, lr, max_english_sentence_length, max_spanish_sentence_length, num_encoder_tokens, num_decoder_tokens, encoder_input_data, decoder_input_data, decoder_target_data,segment_num,load_flag=0,pretrained_path=0):
    #encoder embedding and input layers
    if load_flag == 0:
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
        model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=["accuracy"])
    else:
        model = tf.keras.models.load_model(pretrained_path)


    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2
              )

    model.save(f'{checkpoint_path}/TRAINED_segment{segment_num}.h5')
