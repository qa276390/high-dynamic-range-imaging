# high-dynamic-range-imaging
HDR images have larger dynamic ranges than the ordinary images we see. From recovering the response curve, we could reconstruct the irradiance map in the real scene.


## Project description
There are several steps(shown below) between the scene radiance and the image we see. Starting from taking few photographs in different exposures, we use  [Paul Debevec's method](http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf) to [recover the response curve](#recovering-the-response-curve) by these photographs. After we got the response curve, we are now able to [reconstuct the irradiance map](#reconstruct-the-irradiance-map) also known as HDR images. Also, we developed a [global tone mapping](#global-tone-mapping) algorithm to see the combination of these photographs.

<img src="https://github.com/qa276390/high-dynamic-range-image/blob/master/example/steps.png" height="150"/>


## Algorithms in each steps
### Recovering the Response Curve

Film response curve is a function maps from observed pixel values on an image to the log of exposure values: g(Zij) = ln(Ei) + ln(tj). To recover function g, we implement equation from Debevec
<img src="https://github.com/qa276390/high-dynamic-range-image/blob/master/example/e1.png" height="150"/>

<img src="https://latex.codecogs.com/svg.latex?\Large&space;g" title="g" /> is the unknown response function

<img src="https://latex.codecogs.com/svg.latex?\Large&space;w" title="w" />
is a linear weighting function. g will be less smooth and will fit the data more poorly near extremes (Z=0 or Z=255). Debevec introduces a weighting function to enphasize the smoothness fitting terms toward the middle of the curve.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;t_j" title="t_j" /> is the exposure time

>>E is the unknown radiance

>>Z: is the observed pixel value

>>i is the pixel location index.

>>j is the exposure index

>>P is the total number of exposures.

>This response curve can be used to determine radiance values in any images acquired by the imaging processing associated with g, not just the images used to recover the response curve.

2. High Dynamic Range Radiance Map Construction
>Once the response curve g is recovered, we can construct a radiance map based on equation from Debevec

<img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/e2.png" height="80"/>

>In order to reducing noise in the recovered radiance value, we use all the available exposrues for a particular pixel to computer its radiance based on equation 6 in Debevec. 

### Reconstruct the Irradiance Map

### Global Tone Mapping
>Global tone mapping: In this project, we use gamma correction as global tone mapping. The output image is proportional to the input raised to the power of the inverse of gamma and has pixel value range from 0 to 255.

### Color Adjustment
>In order to construct HDR image to be as closer to input image as possible, we adjust the output image average intensity for each channel (B, G, R) to be the same as template image. In general, we use middle image from image stack as template, which is usually most representative of the ground truth. 

## Result 1
### Original image
<table>
<tr>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample-00.png" /><br>Exposure 1/160 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample-01.png" /><br>Exposure 1/125 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample-02.png" /><br>Exposure 1/80 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample-03.png" /><br>Exposure 1/60 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample-04.png" /><br>Exposure 1/40 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample-05.png" /><br>Exposure 1/15 sec</th>
</tr>
</table>

### HDR image
<img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/output1.png" width="500"/>

## Result 2
### Original image
<table>
<tr>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample2-00.jpg" /><br>Exposure 1/400 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample2-01.jpg" /><br>Exposure 1/250 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample2-02.jpg" /><br>Exposure 1/100 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample2-03.jpg" /><br>Exposure 1/40 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample2-04.jpg" /><br>Exposure 1/25 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample2-05.jpg" /><br>Exposure 1/8 sec</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/sample2-06.jpg" /><br>Exposure 1/3 sec</th>
</tr>
</table>

### HDR image
<img src="https://github.com/vivianhylee/high-dynamic-range-image/raw/master/example/output2.png" width="500"/>




