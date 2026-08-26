# T17 -- Intonation: Averaged F0 Contours

12744/12747 utterances got a usable F0 contour. Averaging across speakers within region is only meaningful because every speaker reads the same sentences -- this figure is the visual argument for parallelism.

## Example contours (sentences with the most cross-region divergence)

![f0_contour_8regions_sentence3.png](figures/f0_contour_8regions_sentence3.png)

*f0_contour_8regions_sentence3.png*

![f0_contour_8regions_sentence237.png](figures/f0_contour_8regions_sentence237.png)

*f0_contour_8regions_sentence237.png*

![f0_contour_8regions_sentence37.png](figures/f0_contour_8regions_sentence37.png)

*f0_contour_8regions_sentence37.png*

![f0_contour_8regions_sentence47.png](figures/f0_contour_8regions_sentence47.png)

*f0_contour_8regions_sentence47.png*

![f0_contour_8regions_sentence57.png](figures/f0_contour_8regions_sentence57.png)

*f0_contour_8regions_sentence57.png*

![f0_contour_8regions_sentence78.png](figures/f0_contour_8regions_sentence78.png)

*f0_contour_8regions_sentence78.png*

## F0 summary features by region

| region     |   f0_range_st |   f0_initial_st |   f0_final_st |   final_slope_st |   n_peaks |
|:-----------|--------------:|----------------:|--------------:|-----------------:|----------:|
| chattogram |        15.968 |           1.009 |        -3.678 |           -0.173 |    20.386 |
| dhaka      |        17.044 |           0.798 |        -4.064 |           -0.202 |    20.689 |
| faridpur   |        16.088 |           0.414 |        -5.956 |           -0.249 |    20.7   |
| khulna     |        18.555 |          -1.024 |        -4.4   |           -0.207 |    19.534 |
| mymensingh |        18.336 |           0.174 |        -4.806 |           -0.216 |    20.934 |
| rajshahi   |        17.394 |          -0.741 |        -5.792 |           -0.23  |    20.696 |
| rangpur    |        19.124 |          -0.542 |        -5.494 |           -0.231 |    21.05  |
| sylhet     |        17.039 |           1.739 |        -4.156 |           -0.217 |    20.646 |

## Contour distance matrix (feeds T18)

|            |   chattogram |   dhaka |   faridpur |   khulna |   mymensingh |   rajshahi |   rangpur |   sylhet |
|:-----------|-------------:|--------:|-----------:|---------:|-------------:|-----------:|----------:|---------:|
| chattogram |      nan     |   1.225 |      1.165 |    1.493 |        1.319 |      1.381 |     1.328 |    1.078 |
| dhaka      |        1.225 | nan     |      1.339 |    1.581 |        1.416 |      1.509 |     1.468 |    1.217 |
| faridpur   |        1.165 |   1.339 |    nan     |    1.514 |        1.284 |      1.26  |     1.276 |    1.138 |
| khulna     |        1.493 |   1.581 |      1.514 |  nan     |        1.645 |      1.678 |     1.635 |    1.509 |
| mymensingh |        1.319 |   1.416 |      1.284 |    1.645 |      nan     |      1.482 |     1.445 |    1.255 |
| rajshahi   |        1.381 |   1.509 |      1.26  |    1.678 |        1.482 |    nan     |     1.476 |    1.363 |
| rangpur    |        1.328 |   1.468 |      1.276 |    1.635 |        1.445 |      1.476 |   nan     |    1.333 |
| sylhet     |        1.078 |   1.217 |      1.138 |    1.509 |        1.255 |      1.363 |     1.333 |  nan     |

![Mean F0 range by region.](figures/f0_range_by_region.png)

*Mean F0 range by region.*

## For the paper

S7.6. One contour figure goes in the main paper as the visual argument for parallelism.
