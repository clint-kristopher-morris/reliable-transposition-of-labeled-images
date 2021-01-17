# reliable-transposition-of-labeled-images
I have been developing and exploring methods to reduce the effort of collecting labeling images for training deep learning models. See my prior work with developing:
*[A deep learning assisted images scraper ]( https://github.com/clint-kristopher-morris/yolo-assisted-image-scrape)
*Implementing SCAN (Learning to Classify Images without Labels)

Another way so expedite the labeling process would be to label image's before augmenting them and transposing both the label and the image.

Another way to expedite the labeling process would be to transpose both the label and image simultaneously. There exist out of the box methods to accomplish this task. One on the best sources being [imgaug](https://imgaug.readthedocs.io/en/latest/source/installation.html).
However, there are some pitfalls to applying this method to your model labels. YOLO labels have the following structure:
```
<object-class> <x>  <y>  <absolute_ width> /<image_width>  <absolute_height> /<image_height>
```
