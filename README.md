# pytorch-GAN
## Description
GANのpytorch実装

### GAN
[papaer link](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- DLによって画像生成を行うモデル
- 2つのモデルを競わせるように学習を行う

## Example
### loss
![loss](https://github.com/Kyou13/pytorch-GAN/blob/master/samples/loss.png)
### Genarated Image
![genaratedImage](https://github.com/Kyou13/pytorch-GAN/blob/master/samples/fake_images_190717030552.png)

## Requirement
- Python 3.7
- pytorch 1.1.0
- torchvision 0.3.0
- Click

## Usage
### Training
```
$ pip install -r requirements.txt
$ python main.py train
# training log saved at ./samples/fake_images-[epoch].png
```

### Generate
```
$ python main.py generate
# saved at ./samples/fake_images_%y%m%d%H%M%S.png
```
