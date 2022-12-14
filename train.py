import os, shutil
import numpy as np
from model import train

#important
resume               = False
epochs_per           = 5
data_path            = "spa_data"
checkpoint_path      = "spa_checkpoints"
final_model_path     = "spa_final_model"

batch_size = 256
latent_dim = 256
learning_rate = 0.01


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
if not resume: clear_dir(checkpoint_path)

params=np.load("spa_data/params.npy", allow_pickle=True)
segments, params = params
segment_size, segment_num, x,y = segments


trained_upto = 0
for segment in range(0,segment_size-trained_upto):
    segment+=trained_upto
    decoder_input_data  = np.load(f'{data_path}/decoder_input_data/segment{segment}.npy', allow_pickle=True)
    encoder_input_data  = np.load(f'{data_path}/encoder_input_data/segment{segment}.npy', allow_pickle=True)
    decoder_target_data = np.load(f'{data_path}/decoder_target_data/segment{segment}.npy', allow_pickle=True)
    pretrained_path = f'{checkpoint_path}/TRAINED_segment{segment-1}.tf'
    max_english_sentence_length, max_spanish_sentence_length, num_encoder_tokens, num_decoder_tokens = params
    train(epochs_per,
          batch_size,
          latent_dim,
          learning_rate,
          encoder_input_data,
          decoder_input_data,
          decoder_target_data,
          segment,
          bool(segment),
          pretrained_path,
          max_english_sentence_length,
          max_spanish_sentence_length,
          num_encoder_tokens,
          num_decoder_tokens,
          checkpoint_path)

gross_epochs = epochs_per*segment_num
print(gross_epochs)
