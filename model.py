
import tensorflow as tf                                                         #model backend
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np                                                              #for messing with tensors
import gc                                                                       #keeps memory usage low
from tensorflow.keras import optimizers                                         #adam
import os

def train(epochs, batch_size, latent_dim, lr, encoder_input_data, decoder_input_data, decoder_target_data,segment_num,load_flag,pretrained_path,max_english_sentence_length, max_spanish_sentence_length, num_encoder_tokens,num_decoder_tokens,checkpoint_path):
    #encoder embedding and input layers
    if load_flag == 0:
        #encoder embedding and input layers
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(num_encoder_tokens, latent_dim,input_length=max_english_sentence_length)(encoder_inputs)
        encoder_stack_h, encoder_state_h, encoder_state_c = LSTM(latent_dim,return_state=True,return_sequences=True)(encoder_embedding)
        encoder_states = [encoder_state_h, encoder_state_c]

        #decoder embedding and dense layer (STOP SETTING THE DENSE NEURON COUNT TO ONE ACE)
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(num_decoder_tokens, latent_dim,input_length=max_spanish_sentence_length)(decoder_inputs)
        decoder_stack_h = LSTM(latent_dim, return_sequences=True)(decoder_embedding, initial_state=encoder_states)

        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])                #finds the dot product between the sequences
        attention = Activation('softmax')(attention)                                    #softmaxes it
        context = dot([attention, encoder_stack_h], axes=[2,1])                         #calculates the context vectors
        decoder_combined_context = concatenate([context, decoder_stack_h])              #combines the context and hidden states

        #dropout = Dropout(0.1)(decoder_combined_context) dropedout was bad here
        decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(decoder_combined_context) #then finds which word is most likely

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

    model.save(f'{checkpoint_path}/TRAINED_segment{segment_num}.tf')
