# Fast neural style transfer

* With VGG16 loss head.
* With UNET fully convolutional net for image generation.

From style images:

Van Gogh   | Picasso
:---------:|:--------:
<img src="./style_images/starry_night.jpg" height="200rm"> | <img src="./style_images/picasso.jpeg" height="300rm">

To train the network with Imagenette:

```bash
pipenv shell
pipenv install
pipenv run python train.py --starry_night --width=<width_int> --height=<height-int>
```

## Examples

* Heidelberg

Original   | Van Gogh  | Picasso
:---------:|:---------:|:---------:
<img src="./data/heidelberg.jpg" height="250rm" > | <img src="./data/heidelberg_styled_vg.png" height="250rm"> | <img src="./data/heidelberg_styled_pic.png" height="250rm">

* Outside scene with friends

Van Gogh | Picasso
:-------:|:-------:
[![mount-vesuvio](https://img.youtube.com/vi/ZjJtOnqJqIg/0.jpg)](https://www.youtube.com/watch?v=ZjJtOnqJqIg) | [![mount-vesuvio](https://img.youtube.com/vi/NzKcvEsIu4s/0.jpg)](https://www.youtube.com/watch?v=NzKcvEsIu4s)

* Mount Vesuvio's inside

Van Gogh | Picasso
:-------:|:-------:
[![outside-scene-with-friends](https://img.youtube.com/vi/xirnt_-sChI/0.jpg)](https://www.youtube.com/watch?v=xirnt_-sChI) | [![outside-scene-with-friends](https://img.youtube.com/vi/lfbySLIlNUk/0.jpg)](https://www.youtube.com/watch?v=lfbySLIlNUk)


### References

* Original source [J.C.Johnson](https://github.com/jcjohnson/fast-neural-style#models-from-the-paper).
* A residual PyTorch implementation [E. Linder-Norén](https://github.com/eriklindernoren/Fast-Neural-Style-Transfer)

## @Alex