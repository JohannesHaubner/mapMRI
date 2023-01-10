import json
import os
import pathlib

import nibabel
import numpy
import numpy as np

from nibabel.affines import apply_affine

image2 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/freesurfer/abby/mri/wmparc.mgz"
image2 = nibabel.load(image2)


reg = [[9.800471067428589e-01, -1.910823285579681e-01, -5.472708866000175e-02, -3.218185424804688e+00],
    [1.941456347703934e-01, 9.792850017547607e-01, 5.751936882734299e-02, -9.298248291015625e-01],
    [4.260246083140373e-02, -6.699671596288681e-02, 9.968433976173401e-01, 4.362060546875000e+00],
    [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.000000000000000e+00]]


# erniedti = [[2.433079481124878e+00, -5.742774009704590e-01, -1.839355193078518e-02, 4.073604583740234e+01  ],
# [-9.531639516353607e-03, 3.967462480068207e-02, -2.499670028686523e+00, 1.875154418945312e+02 ],
# [-5.744929909706116e-01, -2.432824850082397e+00, -3.642201423645020e-02, 2.668685302734375e+02],
# [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.000000000000000e+00 ]]

# erniedti = np.array(erniedti)
# print(np.round(erniedti),1)
# print(np.round(np.linalg.inv(erniedti)), 1)

reg = np.array(reg)
# print(reg)

def fun(x):
    return x # numpy.linalg.inv(x)


def fun2(x):
    return np.matmul(fun(x), (reg))

# print("Image 2 vox2ras_tkr")
# print(np.round(fun(image2.header.get_vox2ras_tkr())))
print("image2 vox2ras     ")
print(np.round(fun(image2.header.get_vox2ras())))


print("wmprac * reg ")
print(np.round(fun2(image2.header.get_vox2ras_tkr())))
print(np.round(fun2(image2.header.get_vox2ras())))



image2 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/registered/abbytoernie.mgz"
image2 = nibabel.load(image2)

print("Abbytoernie")
# print("Image 2 vox2ras_tkr")
# print(np.round(fun(image2.header.get_vox2ras_tkr())))
print("image 2 vox2ras     ")
print(np.round(fun(image2.header.get_vox2ras())))

image2 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/freesurfer/ernie/mri/wmparc.mgz"
image2 = nibabel.load(image2)

print("Ernie")
# print("Image 2 vox2ras_tkr")
# print(np.round(fun(image2.header.get_vox2ras_tkr())))
print("image 2 vox2ras     ")
print(np.round(fun(image2.header.get_vox2ras())))
