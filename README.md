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
-  1 : Using Pretrained MCCNN cost + Cost Volume Filtering + Local Expansion for refining (very slow)
-  2 : Using MCCNN cost trained by us + Cost Volume Filtering (slow)
-  3 : Using MCCNN cost trained by us + Cost Volume Filtering + Local Expansion for refining (very slow)
  
## Evaluate  
  python3 eval_middleBury.py
  
## Result
<p align="center"><img width="50%" height="50%" src="https://github.com/b03901165Shih/2018-CV-Final-Project/blob/master/result/cones.png" /></p>

##### The oringal data for this challenge is not provided due to license reasons.
