# reliable-transposition-of-labeled-images
I have been developing and exploring methods to reduce the effort of collecting labeling images for training deep learning models. See my prior work with developing:
* [A deep learning assisted images scraper ]( https://github.com/clint-kristopher-morris/yolo-assisted-image-scrape)
* Implementing SCAN (Learning to Classify Images without Labels)

Another way to expedite the labeling process would be to transpose both the label and image simultaneously. There exist out of the box methods to accomplish this task. One on the best sources being [imgaug](https://imgaug.readthedocs.io/en/latest/source/installation.html).
However, there are some pitfalls to applying this method to your model labels. The issue with current models is that what lies within the label in a black box with respect to the augmenting model. This makes it extremely difficult to rotate an image and to successfully reconstruct a new label, particularly with the item within the shape has dissimilar lengths and widths. 
For example, in the image below it would be impossible for an augmentation model to know that the item of interest’s length and width are actually much more dissimilar than presented by the label. When rotating and reconstructing a label, the label must translate and scale to the items adjusting new orientation which, can also be seen below.


![phi]( https://i.ibb.co/MN5ZKsv/eg.png)

YOLO labels have the following structure:
```
<object-class> <x>  <y>  <absolute_ width> /<image_width>  <absolute_height> /<image_height>
```
I am studying the side profile of cars traveling on the interstate, the cars move along an inclined surface. Therefore, Φ is known and approximately constant at all locations.

![phi]( https://i.ibb.co/GJM5Txd/uphill600.png)

### Step: 1
Φ can be leveraged to extract information about contents within the label. The true length and width of the vehicle can be found as follows:

![phiall](https://i.ibb.co/sWmBDtN/phiall500.png)

* Eq1: Label_Width = Vehicle_ Height * sin(Φ) + Vehicle_ Width * cos (Φ)
* Eq2: Label_ Height = Vehicle_ Height * cos(Φ) + Vehicle_ Width * sin (Φ)

### Step: 2
Find the true points of the vehicle, convert to polar space then translate the image by a random value θ.

* x' = x * cos (Φ) - y * sin(Φ)
* y' = y * cos (Φ) + x * sin(Φ)

### Step: 3
Reconstruct a new label by accounting for the effects of both θ and Φ on both the length or width.

![](https://i.ibb.co/vLGLCdS/final500.png)


In the figure below, the blue bbox is the original label and red bbox accounts for the true size of the vehicle (step: 1). 
While, the yellow bbox displays what it would look like if you only rotated the points from the first image (step: 2). 
The white box shows how the function accounts for θ and Φ (step: 3).


![](https://i.ibb.co/YP5PyPr/output650-H.png)

### How to Implement This Method:

```
im_aug_transpose_labels(constant_angle, num_im, angle, blur, color_var, path='data/sorted', outfile='data/aug_data')

```
* constant_angle: the angle of the road.
* num_im: number of augmentations you want.
* angle: absolute max rotation angle


### TL;DR
If you are labeling images with a predictable or consistent rotation rotate your images prior to labeling, if you already have labeled images with a consistent skew you can use a method like this to develop a more effective dataset.



