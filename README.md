# Eternal_Cat

# Colab command:

#Change data_path in const.py
data_path = os.path.join(PARENT_PATH, 'drive/Shared drives/Eternal_Cat/data')

#drive mount
from google.colab import drive
drive.mount('/content/drive')

#train
!python preprocess/filtering_to_dataset.py
!python res_AE/preprocess.py
!python preprocess/train_to_idx.py
!python res_AE/train.py

#valset_to_hot
!python res_AE/valset_to_hot.py
