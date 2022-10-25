import os, shutil


global data_path            = "~/env/spa_data"
global net_epochs           = 5000
global checkpoint_path      = "~/env/spa_checkpoints"
global final_model_path     = "~/env/spa_final_model"

def clear_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
clear_dir(checkpoint_path)
