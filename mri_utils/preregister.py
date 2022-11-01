import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.ndimage import affine_transform
import nibabel
from nibabel.affines import apply_affine
from scipy.spatial.transform import Rotation as R

path = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/mridata_3d/"

input_image = nibabel.load(path + "091_cropped.mgz")
input_affine = input_image.affine
input_image = input_image.get_fdata()

target_image = nibabel.load(path + "205_cropped.mgz")
target_image = target_image.get_fdata()

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

dx, dy, dz = 1, 1, 1

# rot = np.array([
#     [  1,    0,    0],
#     [   0,    1,   0],
#     [   0,   0,    0]])
# b = np.array([0, 0, 0])

affine = np.array([
    [1, 0, 0, dx],
    [0, 1, 0, dy],
    [0, 0, 1, dz],
    [0, 0, 0, 1]
    ])


# input_transformed = register(affine, rot=None, b=None)

iterk = 0

def callback(xk):

    global iterk
    iterk += 1

    print(iterk, mismatch(xk))


def assemble_affine(a_ij):

    rot = a_ij[:-3]

    # breakpoint()
    # print(rot.shape)

    rot = R.from_euler('zyx', rot, degrees=True).as_matrix()

    # rot = rot.reshape(3,3)
    # r = R.from_euler('zyx', [90, 45, 30], degrees=True)

    b = a_ij[-3:]

    return rot, b

def mismatch(a_ij, *args):

    rot, b = assemble_affine(a_ij)

    input_transformed = register(aff=None, rot=rot, b=b)

    return np.mean(np.abs(input_transformed - target_image) ** 2)

# x0 = np.concatenate((np.eye(3).flatten(), np.zeros(3)))

x0 = np.zeros(6) + 1.

print("Initial mismatch:", mismatch(x0))

res = minimize(fun=mismatch, x0=x0, args={"maxiter": 1000, "maxfun": 1000}, callback=callback, method="L-BFGS-B")

rot, b = assemble_affine(res.x)

input_transformed = register(aff=None, rot=rot, b=b)


nibabel.save(nibabel.Nifti1Image(input_transformed, affine=input_affine), path + "091registeredto205.mgz")
np.save(path + "optimization_result.npy", arr=res.x)
print("final mismatch:", mismatch(res.x))
print(assemble_affine(res.x))