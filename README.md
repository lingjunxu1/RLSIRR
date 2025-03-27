# Pixel-wise Single Image Reflection Removal Method Based on Reinforcement Learning(ICME-2025)

### Dependencies
* Python3
* PyTorch>=1.0
* NVIDIA GPU+CUDA
* Please ensure that Pytorch can use the GPU normally, and other dependencies do not have high version requirements, you can install by "pip install package-name"

### Data Preparation
#### Training dataset
* 7,643 images from the
  [Pascal VOC dataset(Syn_CEIL)](http://host.robots.ox.ac.uk/pascal/VOC/), center-cropped as 224 x 224 slices to synthesize training pairs. And move it to ```local path```
* 13,700 pair images from the [Berkeley synthetic dataset(Syn_zhang)](https://drive.google.com/drive/folders/1P9xc9vVxk2bbVGhvIwi37MxJuXGf-66i). And move it to```local path```
* 89 real-world training pairs provided by [Zhang *et al.*(Real89)](https://github.com/ceciliavision/perceptual-reflection-removal). And move it to```local path```
* 200 real-world training pairs provided by [IBCLN(Nature200)](https://github.com/JHL-HUST/IBCLN) (In our training setting 2, &dagger; labeled in our paper). And move it to```local path```
#### Testing dataset
* 20 real testing pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal); And move it to ```local path```
* 20 real testing pairs provided by [Li *et al.*](https://github.com/JHL-HUST/IBCLN); And move it to ```local path```
* 454 real testing pairs from [SIR^2 dataset](https://sir2data.github.io/), containing three subsets (i.e., Objects (200), Postcard (199), Wild (55)). Move them separately to their corresponding folders.

#### SIRR Model
Go to the model repository to download pre trained weights and move them to the corresponding folder under ./baseline
* [ERRNet](https://github.com/Vandermode/ERRNet)
* [Kim et al.](https://github.com/sookim813/Reflection_removal_rendering)
* [YTMT](https://github.com/mingcv/YTMT-Strategy)
* [DSRNet](https://github.com/mingcv/DSRNet/tree/main)
* [Zhu et al.](https://github.com/zhuyr97/Reflection_RemoVal_CVPR2024)

### Usage
**Stage I**: `python trainSeg.py --datadirSyn ["Pascal VOC dataset path"] --realData ["Real89 and Nature200 path"] --savePath ["path for save trained seg models"] --policyHid ["Dimension of policy network hidden layer,such as 1024,1024/2 et al."]` 

**Stage II**: `python trainRL.py --datadirSyn ["Berkeley synthetic dataset path"] --realData ["Real89 and Nature200 path"] --segModelPath ["path for use trained seg ckpts"] --savePath ["path for save trained RL model"] --policyHid ["Dimension of policy network hidden layer,such as 1024,1024/2 et al."]` Training Policy Network

**Stage III**: `python finetuning.py --datadirSyn ["Berkeley synthetic dataset path"] --realData ["Real89 and Nature200 path"] --segModelPath ["path for use trained seg ckpts"] --rlModelPath ["path for use trained RL ckpts"] --savePath ["path for save finetuning seg model"] --policyHid ["Dimension of policy network hidden layer,such as 1024,1024/2 et al."]` Fine tuning
#### Testing 
You can put the results of the model inference into folder ```./modelResult``` and then run:

`python test.py --SIRR ["Test model name,  such as ERRNet、Kim、YTMT、DSRNetl and RDRRNet et al."] --maxStep 5 --segModel ["Save folder for segmentation model"] --rLModel["Path for saving policy models"] --dataPath ["Save the path of the test dataset using datasetName/blend, T to store the data pairs"] --resultDir ["Path to save results"] --policyHid ["Dimension of policy network hidden layer,such as 1024,1024/2 et al."] `
