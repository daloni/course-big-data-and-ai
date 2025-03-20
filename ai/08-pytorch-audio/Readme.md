# Audio classification with ResNet

Dataset: https://github.com/karolpiczak/ESC-50

## Transform audio to images

```bash
cd src
python audio_to_image.py
```

## Classify images in different folders

```bash
cd src
python classify_images.py
```

## Create a model

Execute:

```bash
cd src
python train.py
```

## Test your model

Execute:

```bash
cd src
python test.py
```
