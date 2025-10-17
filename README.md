# ProtoPNet-Pytorch 

Protopnet implementation from scratch. The augmentation is not really needed to get close to 80% acc and is too much of a hassle, though the script is there. 

Setup. 
1. Conda environment
```
conda create -p ./protopnet-pytorch python=3.13 -y
pip install torch numpy opencv-python matplotlib torchvision
```

2. Download CUB_200_2011 dataset, unzip it in `datasets/`, and then run the `crop_and_split.py` script to create `cub200_cropped` with a train/test split. 
```
(protopnet)➜  datasets git:(master) ✗ tree -L 1  
.
├── crop_and_split.py
├── CUB_200_2011 
├── cub200_cropped (should contain train_cropped, test_cropped)
├── img_aug.py
├── __init__.py

```

3. If you want train set augmentation. The augmentation script sometimes crashes due to cropping (after rotation) leading to some overflow error. Just rerun it in this case. 
```
pip install augmentor  # optional if you want to augment train set 
python datasets/img_aug.py
```

4. Train and evaluate. You can adjust num epochs here. It is trained with resnet34 backbone and it is always class specific. 
```
python main.py 
```

Initial prototypes are pretty trash, with 1% acc. 

![image](saved/initial_prototypes/prototype_4.png)

Original Repo: https://github.com/cfchen-duke/ProtoPNet

