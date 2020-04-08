# high-dynamic-range-imaging
This is a python project from recovering the response curve to reconstruct the irradiance map(HDR) in the real scene.

- [Project Description](#project-description)
- [Algorithm](#algorithm)
- [Usage](#usage)
- [Results](#resuls)

## Project Description
There are several steps(shown below) between the scene radiance and the image we see. Starting from taking few photographs in different exposures, we use  [Paul Debevec's method](http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf) to [recover the response curve](#recovering-the-response-curve) by these photographs. After we got the response curve, we are now able to [reconstuct the irradiance map](#reconstruct-the-irradiance-map) also known as HDR images. Also, we developed a [global tone mapping](#global-tone-mapping) algorithm to see the combination of these photographs.

<img src="https://github.com/qa276390/high-dynamic-range-image/blob/master/example/steps.png" height="150"/>


## Algorithms 
### Recovering the Response Curve

The function of response curve is a general term for the many steps shown above. To recover from pixel values record on image to radiance map we need a function g which is:  <img src="https://latex.codecogs.com/svg.latex?g(Z_i_j)&space;=&space;ln(E_i)&space;&plus;&space;ln(t_j)" title="g(Z_i_j) = ln(E_i) + ln(t_j)" /></a>.

<img  align="center" src="https://github.com/qa276390/high-dynamic-range-image/blob/master/example/eq1.png" height="50"/>

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;g" title="g" /> is the unknown response function

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;w" title="w" /> is a linear weighting function. g will be less smooth and will fit the data more poorly near extremes (Z=0 or Z=255). Debevec introduces a weighting function to enphasize the smoothness fitting terms toward the middle of the curve.

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;t_j" title="t_j" /> is the exposure time in index <img src="https://latex.codecogs.com/svg.latex?\Large&space;j" title="j" />

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;E_i" title="E_i" /> is the unknown radiance in different pixel <img src="https://latex.codecogs.com/svg.latex?\Large&space;i" title="i" />

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;Z_i_j" title="Z_i_j" />: is the observed value in pixel <img src="https://latex.codecogs.com/svg.latex?\Large&space;i" title="i" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;j" title="j" /> exposure time

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;i" title="i" /> is the pixel location index, <img src="https://latex.codecogs.com/svg.latex?\Large&space;j" title="j" /> is the exposure index and **P** is the total number of exposures.

### Reconstruct the Irradiance Map
And now we can reconstruct the irradiance map by the formula below.
<img src="https://latex.codecogs.com/svg.latex?ln(E_i)&space;=&space;\frac{\sum_{P}^{j=1}w(Z_i_j)(g(Z_i_j)-ln(\Delta&space;t_j)))&space;}{\sum_{P}^{j=1}w(Z_i_j)}" title="ln(E_i) = \frac{\sum_{P}^{j=1}w(Z_i_j)(g(Z_i_j)-ln(\Delta t_j))) }{\sum_{P}^{j=1}w(Z_i_j)}" />
After this step, we are able to have the `output.hdr` file.

### Global Tone Mapping
In this part, we implemented a naive global tone mapping algorithm(shown below) to get a image which can match our visual experience.
<a href="https://www.codecogs.com/eqnedit.php?latex=L_d(x,&space;y)&space;=&space;\frac{L_m(x,y)(1&plus;\frac{L_m(x,y)}{L^2_w(x,y)})}{1&plus;L_m(x,y)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?L_d(x,&space;y)&space;=&space;\frac{L_m(x,y)(1&plus;\frac{L_m(x,y)}{L^2_w(x,y)})}{1&plus;L_m(x,y)}" title="L_d(x, y) = \frac{L_m(x,y)(1+\frac{L_m(x,y)}{L^2_w(x,y)})}{1+L_m(x,y)}" />
and now we can have the result.

## Usages



## Results
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




