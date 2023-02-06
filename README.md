# Multiclass-UNet-PyTorch

The code for UNet is adapted from [this repo](https://github.com/milesial/Pytorch-UNet/). Depends on your running method, click to jump to section: [Local Machine](#run-on-local-machine) or [Google Colab](#run-on-google-colab).

## Run on local machine

Activate your virtual environment. Install the requirements by 

```terminal
pip install -r requirements.txt
```

### 1. Data Preparation

Check if there's one labeled mask:
  - Yes: Increase training data by rotation. Then, proceed to [Step 2](#2-training).
  ```terminal
  cd 1_Data
  python data_augmentation.py
  cd ..
  ```
  - No: Annotate using [labelme](https://github.com/wkentaro/labelme/tree/main/examples/semantic_segmentation).

### 2. Training

Model is defined in the training pipeline. The only parameter that changes is load, which is the weight located in checkpoint folder. 

Download link for [default weight](https://drive.google.com/file/d/1UF9rJbvyg_ClmstrehqCaJ8a3e4mz879/view?usp=sharing) and [latest weight](https://drive.google.com/file/d/1O1Wrvex2v8rK4us8k9OiWB72b-yuC71Q/view?usp=sharing). Put the downloaded weight in the checkpoint folder.

```terminal
cd 2_UNet

# default

python train.py --epochs 200 --batch-size 2 --learning-rate 1e-5 --load checkpoint/saved/unet_carvana_scale1.0_epoch2.pth --scale 1

# latest
python train.py --epochs 200 --batch-size 2 --learning-rate 1e-5 --load checkpoint/saved/run10=5.83e-7.pth --scale 1
```

The training should run for multiples times until the loss stop improving, in my case is 10 runs. Below is the result for each run.
```
1. Epoch 198/200 [loss (batch)=0.0946]
2. Epoch 194/200 [loss (batch)=0.0187]
3. Epoch 140/200 [loss (batch)=0.00505]
4. Epoch 191/200 [loss (batch)=0.000222]
5. Epoch 200/200 [loss (batch)=1.97e-5]
6. Epoch 172/200 [loss (batch)=7.48e-6]
7. Epoch 200/200 [loss (batch)=6.71e-6]
8. Epoch 139/200 [loss (batch)=2.39e-6]
9. Epoch 200/200 [loss (batch)=7.09e-7]
10. Epoch 180/200 [loss (batch)=5.83e-7]
No improvement for the next 200 epochs, best was 1.11e-6
```

### 3. Testing

Even you only want to predict a single image, also put it in a folder.

```py
python predict.py --model checkpoint/saved/run10=5.83e-7.pth --scale 1 --input 1_Data/test/*.png
```

### 4. Visualize

To convert binary mask to RGB images you got in step 3.

```py
python mask_to_rgb.py --path <folder_contain_binary_mask>
```

### 5. Object Count

TO-BE-UPDATE

## Run on Google Colab

For Training and Testing only. Data Preparation should be done locally. Upload this folder to your Google Drive, then open "main.ipynb" from Google Colab.

## Note

1. Inference can be run using GPU with 4GB vram (GTX 1650).
2. In case of CUDA running out of memory during inference, try to clear NVIDIA cache.
