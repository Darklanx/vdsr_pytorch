# VDSR PyTorch
[VDSR](https://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf) PyTorch Implementation  
You can use multi-gpus.  
but no multi-scale.  
And you can input gaussian noise to input images.

## Requirement
`torch`  
`torchvision`  
`python-tk` (or `python3-tk`)
`argparse`
`glob`
`pillow`
A `requirement.txt` is also included for convenience.
## Download dataset
Homework dataset is already included.

## Usage
### Training & Evaluation
```
usage: main.py [-h] --dataset DATASET --crop_size CROP_SIZE
               --upscale_factor UPSCALE_FACTOR [--batch_size BATCH_SIZE]
               [--test_batch_size TEST_BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
               [--step STEP] [--clip CLIP] [--weight-decay WEIGHT_DECAY]
               [--cuda] [--threads THREADS] [--gpuids GPUIDS [GPUIDS ...]]
               [--add_noise] [--noise_std NOISE_STD] [--test] [--model PATH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset directory name
  --crop_size CROP_SIZE
                        network input size
  --upscale_factor UPSCALE_FACTOR
                        super resolution upscale factor
  --batch_size BATCH_SIZE
                        training batch size
  --test_batch_size TEST_BATCH_SIZE
                        testing batch size
  --epochs EPOCHS       number of epochs to train for
  --lr LR               Learning Rate. Default=0.001
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=10
  --clip CLIP           Clipping Gradients. Default=0.4
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        Weight decay, Default: 1e-4
  --cuda                use cuda?
  --threads THREADS     number of threads for data loader to use
  --gpuids GPUIDS [GPUIDS ...]
                        GPU ID for using
  --add_noise           add gaussian noise?
  --noise_std NOISE_STD
                        standard deviation of gaussian noise
  --test                test mode
  --model PATH          path to test or resume model
```

#### To reproduce the homework result, please run:

You can modify the batch_size to fit your GPU, the below code runs for 300 epoch to be ensure convergence, but in fact, around 150 epoch should do the work.

```
> python3 main.py --dataset HW --cuda  --upscale_factor 3 --crop_size 256 --batch_size 60 --test_batch_size 16 --epochs 300 --clip 1 --step 20 --lr 1e-2
```

### Sample usage
```
usage: run.py [-h] --input_folder INPUT_FOLDER --model MODEL
              [--output_filename OUTPUT_FILENAME]
              [--scale_factor SCALE_FACTOR] [--cuda]
              [--gpuids GPUIDS [GPUIDS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER
												folder containing testing images
  --model MODEL         model file to use
  --output_filename OUTPUT_FILENAME
                        where to save the output image
  --scale_factor SCALE_FACTOR
                        factor by which super resolution needed
  --cuda                use cuda
  --gpuids GPUIDS [GPUIDS ...]
                        GPU ID for using
```

#### After training the model:

Run the below code and you will find the result in the folder **./submission**

```
> python3 run.py  --scale_factor 3 --model model_epoch_300.pth --folder ./testing_lr_images
```
