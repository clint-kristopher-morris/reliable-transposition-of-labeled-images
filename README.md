# reliable-transposition-of-labeled-images
I have been developing and exploring methods to reduce the effort of collecting labeling images for training deep learning models. See my prior work with developing:
*[A deep learning assisted images scraper ]( https://github.com/clint-kristopher-morris/yolo-assisted-image-scrape)
*Implementing SCAN (Learning to Classify Images without Labels)

Another way to expedite the labeling process would be to transpose both the label and image simultaneously. There exist out of the box methods to accomplish this task. One on the best sources being [imgaug](https://imgaug.readthedocs.io/en/latest/source/installation.html).
However, there are some pitfalls to applying this method to your model labels. The issue with current models is that what lies within the label in a black box with respect to the augmenting model. This makes it extremely difficult to rotate an image and to successfully reconstruct a new label, particularly with the item within the shape has dissimilar lengths and widths. 
For example, in the image below it would be impossible for an augmentation model to know that the item of interest’s length and width are actually much more dissimilar than presented by the label. When rotating and reconstructing a label, the label must translate and scale to the items adjusting new orientation which, can also be seen below.

![phi]( https://i.ibb.co/MN5ZKsv/eg.png)

YOLO labels have the following structure:
```
<object-class> <x>  <y>  <absolute_ width> /<image_width>  <absolute_height> /<image_height>
```
I am studying the side profile of cars traveling on the interstate, the cars move along an inclined surface.


![phi]( https://i.ibb.co/GJM5Txd/uphill600.png)


