Task2_3D adaptive

Adds Adaptive threshold and 1 additional Gaussian smoothing:
Functions
Function # def matrix_adaptive_threshold(matrix), 
Body
# 6_add1: Guassian smoothing of A_T
# 7. Apply adaptive threshold to binarize A_T_sigma_T


Task2_3D_adpt_init

It calculates the porosity of the input 2D image.
When creating the 3D constructs A1 and A2, it forces the randomness of 0-1 to have a % of zeros equal to such porosity
The optimization will happen later, the idea here is not to obtain something exactly like the input 2D image
but to have a better start. It can be used and see if it saves iterations or not.
