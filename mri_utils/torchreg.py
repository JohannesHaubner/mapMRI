import torch
import torchio
import numpy as np
# import scipy
# from scipy.optimize import minimize
# from scipy.ndimage import affine_transform
import nibabel
# from nibabel.affines import apply_affine
#from scipy.spatial.transform import Rotation as R

"""

An attempt to implement to implement the image registration (via affine transfrom) in PyTorch.
The transformation would be differentiable and hence the gradients would be obtained easily.

TODO
FIXME 

"""


path = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/mridata_3d/"

input_image = nibabel.load(path + "091_cropped.mgz")
input_affine = input_image.affine
input_image = torch.unsqueeze(torch.from_numpy(input_image.get_fdata()), dim=0)

target_image = nibabel.load(path + "205_cropped.mgz")
target_image = torch.unsqueeze(torch.from_numpy(target_image.get_fdata()), dim=0)

print(target_image.shape)

# input_image = np.zeros((16, 16, 16))
# input_image[4,4,4] = 1
# target_image = np.zeros((16, 16, 16))
# target_image[10,5,5] = 1

def register(aff, rot, b):
    if aff is not None:
        rot = aff[:-1, :-1]
        b = rot[:-1, -1]

    # print(rot, b)

    #  np.dot(matrix, o) + offset.  
    traff = affine_transform(input_image, matrix=rot, offset=b)
    # traff = affine_transform(affine, input_image).T
    return traff



def callback(xk):

    global iterk
    iterk += 1

    print(iterk, mismatch(xk))


def mismatch(degs, b):

    aff = torchio.Affine(scales=torch.ones(3), degrees=degs, translation=b)


    transformed_tensor = aff(input_image)

    return torch.mean(torch.abs(transformed_tensor - target_image))

# x0 = np.concatenate((np.eye(3).flatten(), np.zeros(3)))

degs = torch.zeros(3)

degs = torch.nn.Parameter(degs)
degs.requires_grad = True


b = torch.zeros(3)
b = torch.nn.Parameter(b)
b.requires_grad = True

print("Initial mismatch:", mismatch(degs, b))

optimizer = torch.optim.LBFGS(params=[degs, b])

def closure():
    optimizer.zero_grad()
    loss = mismatch(degs, b)
    loss.backward()
    return loss

optimizer.step(closure)



# input_transformed = register(aff=None, rot=rot, b=b)


# nibabel.save(nibabel.Nifti1Image(input_transformed, affine=input_affine), path + "091registeredto205.mgz")
# np.save(path + "optimization_result.npy", arr=res.x)
# print("final mismatch:", mismatch(res.x))
# print(assemble_affine(res.x))