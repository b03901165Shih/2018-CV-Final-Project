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
