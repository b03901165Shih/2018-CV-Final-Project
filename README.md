# 2018-CV-Final-Project

## Dependencies
- Pymaxflow (https://github.com/pmneila/PyMaxflow)
- Cupy 2.2.0
- Chainer 3.2.0 (GPU version)
- Keras 2.0.9

## Usage 
  python3 main.py --setting \<option>
  
 \<option>:
-  0 : Using Pretrained MCCNN cost + Cost Volume Filtering
-  1 : Using Pretrained MCCNN cost + Cost Volume Filtering + Local Expansion for refining (very slow/ better performance)
-  2 : Using MCCNN cost trained by us + Cost Volume Filtering (slow)
-  3 : Using MCCNN cost trained by us + Cost Volume Filtering + Local Expansion for refining (very slow/ better performance)
  
## Evaluate  
  python3 eval_middleBury.py
  
## Result (option 0/option 1)
Tsukuba:
<p align="left">
  <img width="40%" height="40%" src="https://github.com/b03901165Shih/2018-CV-Final-Project/blob/master/result/tsukuba.png" />
  <img width="40%" height="40%" src="https://github.com/b03901165Shih/2018-CV-Final-Project/blob/master/result_option1/tsukuba.png" />
</p>

Venus:
<p align="left">
  <img width="40%" height="40%" src="https://github.com/b03901165Shih/2018-CV-Final-Project/blob/master/result/venus.png" />
  <img width="40%" height="40%" src="https://github.com/b03901165Shih/2018-CV-Final-Project/blob/master/result_option1/venus.png" />
</p>

Cones:
<p align="left"
><img width="40%" height="40%" src="https://github.com/b03901165Shih/2018-CV-Final-Project/blob/master/result/cones.png" />
><img width="40%" height="40%" src="https://github.com/b03901165Shih/2018-CV-Final-Project/blob/master/result_option1/cones.png" />
</p>

Teddy:
<p align="left">
  <img width="40%" height="40%" src="https://github.com/b03901165Shih/2018-CV-Final-Project/blob/master/result/teddy.png" />
  <img width="40%" height="40%" src="https://github.com/b03901165Shih/2018-CV-Final-Project/blob/master/result_option1/teddy.png" />
</p>


##### The oringal data for this challenge is not provided due to license reasons.

## References
[1] A. Hosni, C. Rhemann, M. Bleyer, C. Rother, and M. Gelautz. Fast cost-volume filtering for visual correspondence and beyond. IEEE Trans. Pattern Anal. Mach. Intell., 35(2):504–511, 2013.

[2] Zbontar, Jure, and Yann LeCun. "Stereo matching by training a convolutional neural network to compare image patches." Journal of Machine Learning Research 17.1-32 (2016): 2.

[3] Taniai, Tatsunori, et al. "Continuous 3D label stereo matching using local expansion moves." IEEE transactions on pattern analysis and machine intelligence 40.11 (2018): 2725-2739.

[4] Boykov, Yuri, Olga Veksler, and Ramin Zabih. "Fast approximate energy minimization via graph cuts." IEEE Transactions on pattern analysis and machine intelligence 23.11 (2001): 1222-1239.

[5] MCCNN code reference: https://github.com/t-taniai/mc-cnn-chainer
