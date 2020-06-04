from __future__ import print_function

import numpy as np
from PIL import Image

from cpython cimport bool
cimport numpy as np

# This is a port of https://pastebin.com/cbH8ZQQT
# *** Super-xBR code begins here - MIT LICENSE ***

# *******  Super XBR Scaler  *******
#
# Copyright (c) 2016 Hyllian - sergiogdb@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cdef int rb(col):
    """Get red byte"""
    return col >> 0 & 0xff

cdef int gb(col):
    """Get green byte"""
    return col >> 8 & 0xff

cdef int bb(col):
    """Get blue byte"""
    return col >> 16 & 0xff

cdef int ab(col):
    """Get alpha byte"""
    return col >> 24 & 0xff

# weights
cdef double wgt1 = 0.129633
cdef double wgt2 = 0.175068
cdef double w1 = -wgt1
cdef double w2 = wgt1 + 0.5
cdef double w3 = -wgt2
cdef double w4 = wgt2 + 0.5

cdef double df(double x, double y):
    """Get absolute difference of two values"""
    return abs(x - y)

cdef double clamp(double x, double low, double high):
    """Clamp x between low and high"""
    return max(min(x, high), low)

'''
                         P1
|P0|B |C |P1|         C     F4          |a0|b1|c2|d3|
|D |E |F |F4|      B     F     I4       |b0|c1|d2|e3|   |e1|i1|i2|e2|
|G |H |I |I4|   P0    E  A  I     P3    |c0|d1|e2|f3|   |e3|i3|i4|e4|
|P2|H5|I5|P3|      D     H     I5       |d0|e1|f2|g3|
                      G     H5
                         P2

sx, sy   
-1  -1 | -2  0   (x+y) (x-y)    -3  1  (x+y-1)  (x-y+1)
-1   0 | -1 -1                  -2  0
-1   1 |  0 -2                  -1 -1
-1   2 |  1 -3                   0 -2

 0  -1 | -1  1   (x+y) (x-y)      ...     ...     ...
 0   0 |  0  0
 0   1 |  1 -1
 0   2 |  2 -2
 
 1  -1 |  0  2   ...
 1   0 |  1  1
 1   1 |  2  0
 1   2 |  3 -1
 
 2  -1 |  1  3   ...
 2   0 |  2  2
 2   1 |  3  1
 2   2 |  4  0
 
'''

# noinspection PyUnresolvedReferences
cdef int diagonal_edge(np.ndarray[double, ndim=2] mat, tuple wp):
    cdef int dw1 = (
            wp[0] * (df(mat[0, 2], mat[1, 1]) + df(mat[1, 1], mat[2, 0]) +
                     df(mat[1, 3], mat[2, 2]) + df(mat[2, 2], mat[3, 1])) +
            wp[1] * (df(mat[0, 3], mat[1, 2]) + df(mat[2, 1], mat[3, 0])) +
            wp[2] * (df(mat[0, 3], mat[2, 1]) + df(mat[1, 2], mat[3, 0])) +
            wp[3] * df(mat[1, 2], mat[2, 1]) +
            wp[4] * (df(mat[0, 2], mat[2, 0]) + df(mat[1, 3], mat[3, 1])) +
            wp[5] * (df(mat[0, 1], mat[1, 0]) + df(mat[2, 3], mat[3, 2]))
    )

    cdef int dw2 = (
            wp[0] * (df(mat[0, 1], mat[1, 2]) + df(mat[1, 2], mat[2, 3]) +
                     df(mat[1, 0], mat[2, 1]) + df(mat[2, 1], mat[3, 2])) +
            wp[1] * (df(mat[0, 0], mat[1, 1]) + df(mat[2, 2], mat[3, 3])) +
            wp[2] * (df(mat[0, 0], mat[2, 2]) + df(mat[1, 1], mat[3, 3])) +
            wp[3] * df(mat[1, 1], mat[2, 2]) +
            wp[4] * (df(mat[1, 0], mat[3, 2]) + df(mat[0, 1], mat[2, 3])) +
            wp[5] * (df(mat[0, 2], mat[1, 3]) + df(mat[2, 0], mat[3, 1]))
    )

    return dw1 - dw2

# Super-xBR scaling
cdef bytes scale_from_buffer(bytes buffer, int w, int h, bool display_progress=True):
    cdef np.ndarray[np.uint32_t, ndim=1] data = np.frombuffer(buffer, np.uint32)

    # only scaling by a factor of 2 is supported
    cdef int f = 2

    cdef int outw = w * f
    cdef int outh = h * f

    # progress feedback
    cdef int progress_current = 0
    cdef int progress_total = (w * h * 2) + (outh * outw)

    # output image buffer array
    cdef np.ndarray[np.uint32_t] out = np.empty(outw * outh, np.uint32)

    cdef tuple wp = (2, 1, -1, 4, -1, 1)

    cdef np.ndarray[double, ndim=2] r = np.empty((4, 4), float)
    cdef np.ndarray[double, ndim=2] g = np.empty((4, 4), float)
    cdef np.ndarray[double, ndim=2] b = np.empty((4, 4), float)
    cdef np.ndarray[double, ndim=2] a = np.empty((4, 4), float)
    cdef np.ndarray[double, ndim=2] luma = np.empty((4, 4), float)

    # declare cython var types for much improved performance
    cdef np.uint32_t sample
    cdef int cx, cy, csx, csy
    cdef double min_r_sample, min_g_sample, min_b_sample, min_a_sample
    cdef double max_r_sample, max_g_sample, max_b_sample, max_a_sample
    cdef double rf, gf, bf, af
    cdef int ri, gi, bi, ai

    # first pass
    for y in range(0, outh, 2):
        for x in range(0, outw, 2):

            # central pixels on original image
            cx = x // f
            cy = y // f

            # sample supporting pixels in original image
            for sx in range(-1, 3):
                for sy in range(-1, 3):
                    # clamp pixel locations
                    csy = int(clamp(sy + cy, 0, h - 1))
                    csx = int(clamp(sx + cx, 0, w - 1))

                    # sample & add weight components
                    sample = data[csy * w + csx]
                    r[sx + 1, sy + 1] = rb(sample)
                    g[sx + 1, sy + 1] = gb(sample)
                    b[sx + 1, sy + 1] = bb(sample)
                    a[sx + 1, sy + 1] = ab(sample)
                    luma[sx + 1, sy + 1] = (0.2126 * r[sx + 1, sy + 1] +
                                            0.7152 * g[sx + 1, sy + 1] +
                                            0.0722 * b[sx + 1, sy + 1])

            min_r_sample = min(r[1, 1], r[1, 2], r[2, 1], r[2, 2])
            min_g_sample = min(g[1, 1], g[1, 2], g[2, 1], g[2, 2])
            min_b_sample = min(b[1, 1], b[1, 2], b[2, 1], b[2, 2])
            min_a_sample = min(a[1, 1], a[1, 2], a[2, 1], a[2, 2])

            max_r_sample = max(r[1, 1], r[1, 2], r[2, 1], r[2, 2])
            max_g_sample = max(g[1, 1], g[1, 2], g[2, 1], g[2, 2])
            max_b_sample = max(b[1, 1], b[1, 2], b[2, 1], b[2, 2])
            max_a_sample = max(a[1, 1], a[1, 2], a[2, 1], a[2, 2])

            d_edge = diagonal_edge(luma, wp)

            # generate and write result
            if d_edge <= 0:
                rf = w1 * (r[0, 3] + r[3, 0]) + w2 * (r[1, 2] + r[2, 1])
                gf = w1 * (g[0, 3] + g[3, 0]) + w2 * (g[1, 2] + g[2, 1])
                bf = w1 * (b[0, 3] + b[3, 0]) + w2 * (b[1, 2] + b[2, 1])
                af = w1 * (a[0, 3] + a[3, 0]) + w2 * (a[1, 2] + a[2, 1])
            else:
                rf = w1 * (r[0, 0] + r[3, 3]) + w2 * (r[1, 1] + r[2, 2])
                gf = w1 * (g[0, 0] + g[3, 3]) + w2 * (g[1, 1] + g[2, 2])
                bf = w1 * (b[0, 0] + b[3, 3]) + w2 * (b[1, 1] + b[2, 2])
                af = w1 * (a[0, 0] + a[3, 3]) + w2 * (a[1, 1] + a[2, 2])

            # anti-ringing, clamp
            rf = clamp(rf, min_r_sample, max_r_sample)
            gf = clamp(gf, min_g_sample, max_g_sample)
            bf = clamp(bf, min_b_sample, max_b_sample)
            af = clamp(af, min_a_sample, max_a_sample)

            ri = int(clamp(np.ceil(rf), 0, 255))
            gi = int(clamp(np.ceil(gf), 0, 255))
            bi = int(clamp(np.ceil(bf), 0, 255))
            ai = int(clamp(np.ceil(af), 0, 255))

            out[y * outw + x] = out[y * outw + x + 1] = out[(y + 1) * outw + x] = data[cy * w + cx]
            out[(y + 1) * outw + x + 1] = (ai << 24) | (bi << 16) | (gi << 8) | ri

            if display_progress:
                progress_current += 1
                print('({}/{})'.format(progress_current, progress_total), end='\r')

    # second pass
    wp = (2, 0, 0, 0, 0, 0)

    for y in range(0, outh, 2):
        for x in range(0, outw, 2):

            # sample supporting pixels in original image
            for sx in range(-1, 3):
                for sy in range(-1, 3):
                    # clamp pixel locations
                    csy = int(clamp(sx - sy + y, 0, f * h - 1))
                    csx = int(clamp(sx + sy + x, 0, f * w - 1))

                    # sample & add weighted components
                    sample = out[csy * outw + csx]
                    r[sx + 1, sy + 1] = rb(sample)
                    g[sx + 1, sy + 1] = gb(sample)
                    b[sx + 1, sy + 1] = bb(sample)
                    a[sx + 1, sy + 1] = ab(sample)
                    luma[sx + 1, sy + 1] = (0.2126 * r[sx + 1, sy + 1] +
                                            0.7152 * g[sx + 1, sy + 1] +
                                            0.0722 * b[sx + 1, sy + 1])

            min_r_sample = min(r[1, 1], r[1, 2], r[2, 1], r[2, 2])
            min_g_sample = min(g[1, 1], g[1, 2], g[2, 1], g[2, 2])
            min_b_sample = min(b[1, 1], b[1, 2], b[2, 1], b[2, 2])
            min_a_sample = min(a[1, 1], a[1, 2], a[2, 1], a[2, 2])

            max_r_sample = max(r[1, 1], r[1, 2], r[2, 1], r[2, 2])
            max_g_sample = max(g[1, 1], g[1, 2], g[2, 1], g[2, 2])
            max_b_sample = max(b[1, 1], b[1, 2], b[2, 1], b[2, 2])
            max_a_sample = max(a[1, 1], a[1, 2], a[2, 1], a[2, 2])

            d_edge = diagonal_edge(luma, wp)

            # generate and write result
            if d_edge <= 0:
                rf = w1 * (r[0, 3] + r[3, 0]) + w2 * (r[1, 2] + r[2, 1])
                gf = w1 * (g[0, 3] + g[3, 0]) + w2 * (g[1, 2] + g[2, 1])
                bf = w1 * (b[0, 3] + b[3, 0]) + w2 * (b[1, 2] + b[2, 1])
                af = w1 * (a[0, 3] + a[3, 0]) + w2 * (a[1, 2] + a[2, 1])
            else:
                rf = w1 * (r[0, 0] + r[3, 3]) + w2 * (r[1, 1] + r[2, 2])
                gf = w1 * (g[0, 0] + g[3, 3]) + w2 * (g[1, 1] + g[2, 2])
                bf = w1 * (b[0, 0] + b[3, 3]) + w2 * (b[1, 1] + b[2, 2])
                af = w1 * (a[0, 0] + a[3, 3]) + w2 * (a[1, 1] + a[2, 2])

            # anti-ringing, clamp
            rf = clamp(rf, min_r_sample, max_r_sample)
            gf = clamp(gf, min_g_sample, max_g_sample)
            bf = clamp(bf, min_b_sample, max_b_sample)
            af = clamp(af, min_a_sample, max_a_sample)

            ri = int(clamp(np.ceil(rf), 0, 255))
            gi = int(clamp(np.ceil(gf), 0, 255))
            bi = int(clamp(np.ceil(bf), 0, 255))
            ai = int(clamp(np.ceil(af), 0, 255))

            out[y * outw + x + 1] = (ai << 24) | (bi << 16) | (gi << 8) | ri

            for sx in range(-1, 3):
                for sy in range(-1, 3):
                    # clamp pixel locations
                    csy = int(clamp(sx - sy + 1 + y, 0, f * h - 1))
                    csx = int(clamp(sx + sy - 1 + x, 0, f * w - 1))

                    # sample and add weighted components
                    sample = out[csy * outw + csx]
                    r[sx + 1, sy + 1] = rb(sample)
                    g[sx + 1, sy + 1] = gb(sample)
                    b[sx + 1, sy + 1] = bb(sample)
                    a[sx + 1, sy + 1] = ab(sample)
                    luma[sx + 1, sy + 1] = (0.2126 * r[sx + 1, sy + 1] +
                                            0.7152 * g[sx + 1, sy + 1] +
                                            0.0722 * b[sx + 1, sy + 1])

            d_edge = diagonal_edge(luma, wp)

            # generate and write result
            if d_edge <= 0:
                rf = w1 * (r[0, 3] + r[3, 0]) + w2 * (r[1, 2] + r[2, 1])
                gf = w1 * (g[0, 3] + g[3, 0]) + w2 * (g[1, 2] + g[2, 1])
                bf = w1 * (b[0, 3] + b[3, 0]) + w2 * (b[1, 2] + b[2, 1])
                af = w1 * (a[0, 3] + a[3, 0]) + w2 * (a[1, 2] + a[2, 1])
            else:
                rf = w1 * (r[0, 0] + r[3, 3]) + w2 * (r[1, 1] + r[2, 2])
                gf = w1 * (g[0, 0] + g[3, 3]) + w2 * (g[1, 1] + g[2, 2])
                bf = w1 * (b[0, 0] + b[3, 3]) + w2 * (b[1, 1] + b[2, 2])
                af = w1 * (a[0, 0] + a[3, 3]) + w2 * (a[1, 1] + a[2, 2])

            # anti-ringing, clamp
            rf = clamp(rf, min_r_sample, max_r_sample)
            gf = clamp(gf, min_g_sample, max_g_sample)
            bf = clamp(bf, min_b_sample, max_b_sample)
            af = clamp(af, min_a_sample, max_a_sample)

            ri = int(clamp(np.ceil(rf), 0, 255))
            gi = int(clamp(np.ceil(gf), 0, 255))
            bi = int(clamp(np.ceil(bf), 0, 255))
            ai = int(clamp(np.ceil(af), 0, 255))

            out[(y + 1) * outw + x] = (ai << 24) | (bi << 16) | (gi << 8) | ri

            if display_progress:
                progress_current += 1
                print('({}/{})'.format(progress_current, progress_total), end='\r')

    # third pass
    wp = (2, 1, -1, 4, -1, 1)

    for y in range(outh - 1, -1, -1):
        for x in range(outw - 1, -1, -1):

            for sx in range(-2, 2):
                for sy in range(-2, 2):
                    # clamp pixel locations
                    csy = int(clamp(sy + y, 0, f * h - 1))
                    csx = int(clamp(sx + x, 0, f * w - 1))

                    # sample & add weighted components
                    sample = out[csy * outw + csx]
                    r[sx + 2, sy + 2] = rb(sample)
                    g[sx + 2, sy + 2] = gb(sample)
                    b[sx + 2, sy + 2] = bb(sample)
                    a[sx + 2, sy + 2] = ab(sample)
                    luma[sx + 2, sy + 2] = (0.2126 * r[sx + 1, sy + 1] +
                                            0.7152 * g[sx + 1, sy + 1] +
                                            0.0722 * b[sx + 1, sy + 1])

            min_r_sample = min(r[1, 1], r[1, 2], r[2, 1], r[2, 2])
            min_g_sample = min(g[1, 1], g[1, 2], g[2, 1], g[2, 2])
            min_b_sample = min(b[1, 1], b[1, 2], b[2, 1], b[2, 2])
            min_a_sample = min(a[1, 1], a[1, 2], a[2, 1], a[2, 2])

            max_r_sample = max(r[1, 1], r[1, 2], r[2, 1], r[2, 2])
            max_g_sample = max(g[1, 1], g[1, 2], g[2, 1], g[2, 2])
            max_b_sample = max(b[1, 1], b[1, 2], b[2, 1], b[2, 2])
            max_a_sample = max(a[1, 1], a[1, 2], a[2, 1], a[2, 2])

            d_edge = diagonal_edge(luma, wp)

            # generate and write result
            if d_edge <= 0:
                rf = w1 * (r[0, 3] + r[3, 0]) + w2 * (r[1, 2] + r[2, 1])
                gf = w1 * (g[0, 3] + g[3, 0]) + w2 * (g[1, 2] + g[2, 1])
                bf = w1 * (b[0, 3] + b[3, 0]) + w2 * (b[1, 2] + b[2, 1])
                af = w1 * (a[0, 3] + a[3, 0]) + w2 * (a[1, 2] + a[2, 1])
            else:
                rf = w1 * (r[0, 0] + r[3, 3]) + w2 * (r[1, 1] + r[2, 2])
                gf = w1 * (g[0, 0] + g[3, 3]) + w2 * (g[1, 1] + g[2, 2])
                bf = w1 * (b[0, 0] + b[3, 3]) + w2 * (b[1, 1] + b[2, 2])
                af = w1 * (a[0, 0] + a[3, 3]) + w2 * (a[1, 1] + a[2, 2])

            # anti-ringing, clamp
            rf = clamp(rf, min_r_sample, max_r_sample)
            gf = clamp(gf, min_g_sample, max_g_sample)
            bf = clamp(bf, min_b_sample, max_b_sample)
            af = clamp(af, min_a_sample, max_a_sample)

            ri = int(clamp(np.ceil(rf), 0, 255))
            gi = int(clamp(np.ceil(gf), 0, 255))
            bi = int(clamp(np.ceil(bf), 0, 255))
            ai = int(clamp(np.ceil(af), 0, 255))

            out[y * outw + x] = (ai << 24) | (bi << 16) | (gi << 8) | ri

            if display_progress:
                progress_current += 1
                print('({}/{})'.format(progress_current, progress_total), end='\r')

    if display_progress:
        print()

    return out.tobytes()

# *** Super-xBR code ends here - MIT LICENSE ***

def scale(im: Image.Image, int passes=1, print_progress=False) -> Image.Image:
    """
    Apply the Super-xBR upscale filter to an image.

    :param im: The PIL Image object to apply the filter to. For best results, it should be a small pixel-art image.
    :param passes: The number of times to apply the filter. The image scale is doubled each time the filter is applied.
    :param print_progress: If True, progress will be displayed while the filter is being applied. This is good when
    working with multiple passes or large images.
    :return: A larger image with the filter applied.
    """
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    cdef bytes im_buffer = im.tobytes()
    for p in range(passes):
        if print_progress:
            print('pass {} of {}...'.format(p + 1, passes))
        im_buffer = scale_from_buffer(im_buffer, im.width * (2 ** p), im.height * (2 ** p), print_progress)
    return Image.frombuffer('RGBA', (im.width * (2 ** passes), im.height * (2 ** passes)), im_buffer)
