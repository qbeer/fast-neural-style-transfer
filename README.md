# Fast neural style transfer (in developement)

* With VGG16 loss head.
* With UNET fully convolutional net for image generation.

Current status :

![reconstruction-via-style-transfer](./reco.png)

From style image:

![van-gogh-starry-night](./starry_night.jpg)

Original source [J.C.Johnson](https://github.com/jcjohnson/fast-neural-style#models-from-the-paper).

To train the network with CIFAR-10:

```bash
pipenv shell
pipenv install
pipenv run python train.py
```

## @Alex