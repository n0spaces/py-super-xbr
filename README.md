# py-super-xbr

The [Super-xBR pixel-art scaling algorithm](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms#xBR_family)
implemented as a Python module and command line tool.

## Examples

Fawful (Mario & Luigi: Superstar Saga) scaled 2x ([Image source](https://www.spriters-resource.com/fullview/7463/)) <br>
![](examples/fawful_2x_nearest.png "Scaled 2x w/ nearest neighbor")
![](examples/fawful_2x_superxbr.png "Scaled 2x w/ Super-xBR")

Misery (Cave Story) scaled 4x ([Image source](https://www.spriters-resource.com/fullview/34560/)) <br>
![](examples/misery_4x_nearest.png "Scaled 4x w/ nearest neighbor")
![](examples/misery_4x_superxbr.png "Scaled 4x w/ Super-xBR")

## Requirements

Python 3.6 or later. This was tested on Windows and Linux, but should also work on macOS.

## Installation

This isn't uploaded to PyPI yet, so you'll have to build and install this from source.

1. Clone this repository to your system
2. Open a terminal and go to the repository's directory
3. Run `pip install .`

## Usage

### From the command line

Applies the Super-xBR upscale filter to the image `INPUT` and saves it to `OUTPUT`:

```commandline
superxbr [OPTIONS] INPUT OUTPUT
```

##### Options

    positional arguments:
      INPUT       Input image file path
      OUTPUT      Output image file path
    
    optional arguments:
      -h, --help  show this help message and exit
      -p PASSES   Number of times to apply the Super-xBR filter. The image
                  scale is doubled each time the filter is applied. (default: 1)
      -m MODE     Color mode of the output image file. Choices are 1, L, LA, I, P,
                  RGB, RGBA, or CMYK. Note that some color modes are not
                  compatible with some image formats and will raise an exception.
                  (default: RGBA)
      --quiet     Hide progress updates



### In a Python script

Images are scaled using `superxbr.scale()`

```python
from PIL import Image  # Pillow 7.x
from superxbr import superxbr

# Create an Image object
im = Image.open('example.png')

# Apply the Super-xBR scale filter to the image
im_scaled_2x = superxbr.scale(im)

# You can apply the filter multiple times
# The scale is doubled each time the filter is applied
im_scaled_8x = superxbr.scale(im, 3)
```

## Building

Install required packages:

    pip install -r requirements.txt

[superxbr.c](superxbr/superxbr.c) needs to be regenerated if [superxbr.pyx](superxbr/superxbr.pyx) is modified:

    cython superxbr/superxbr.pyx

To compile a python extension module (.pyd or .so file):

    python3 setup.py build_ext --inplace

To build a wheel:

    python3 setup.py bdist_wheel

## License

py-super-xbr is available under the MIT License. The code is based on the
[Super-xBR C++ port](https://pastebin.com/cbH8ZQQT) released by [Hylian](mailto:sergiogdb@gmail.com) also under the MIT
License.

See [LICENSE.txt](LICENSE.txt).

## See also

* [Pixel-art scaling algorithms on Wikipedia](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms)
* [Super-xBR explanation by Hylian](https://drive.google.com/file/d/0B_yrhrCRtu8GYkxreElSaktxS3M/view?pref=2&pli=1)
