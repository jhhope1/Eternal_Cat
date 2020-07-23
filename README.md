# Eternal_Cat

## Colab command:
### 1. Copy res_AE and preprocess to content
Paste yout code into Colab folder named 'content'
### 2. Change data_path in const.py
Replace data_path in the const.py file with:
'''python
data_path = os.path.join(PARENT_PATH, 'drive/Shared drives/Eternal_Cat/data')
'''
### 3. Drive mount
Connect drive with Eternal_cat to Kolab
'''colab
from google.colab import drive
drive.mount('/content/drive')
'''
### 4. Train
After preprocessing, train the neural network.
'''colab
!python preprocess/filtering_to_dataset.py
!python res_AE/preprocess.py
!python preprocess/train_to_idx.py
!python res_AE/train.py
'''
### 5. Valset_to_hot
Make 'results.json'
'''colab
!python res_AE/valset_to_hot.py
'''