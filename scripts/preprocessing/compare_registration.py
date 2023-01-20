import os, pathlib, json, subprocess
# from fenics import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from parse import parse
# from scipy.interpolate import CubicSpline
import argparse
# from scipy.signal import savgol_filter
import nibabel
from dgregister.MRI2FEM import read_image

path = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset/processed/"

os.chdir(path)





from ufl import sign

def abs(x):
    return sign(x) * x

sign = np.sign
abs = np.abs

def heavyside(x):

    return (sign(x) + 1) / 2

def tukey(x, c=4.68):

    a = c ** 2 / 2 * (1 - (1 - x ** 2 / (c ** 2)) **3 )
    b = c ** 2 / 2
    x = abs(x)
    y = a * heavyside(c-x) + b * heavyside(x - c)

    return y

def tukey2(x, c=4.68):

    a = c ** 2 / 2 * (1 - (1 - x ** 2 / (c ** 2)) **3 )

    b = c ** 2 / 2

    y = np.where(np.abs(x) <= c, a, b)


    return y

c=4.68

x = np.linspace(-5, 5,100)

y = tukey(x, c)

plt.plot(x, x**2)
plt.plot(x, y, marker="x", markevery=10)
plt.plot(x, tukey2(x,c=c), color="red")
# plt.show()
# exit()



rigid = "registered/"
affine = "affine_registered/"

viewcommand = "freeview "
viewcommand += "input/abby/abby_brain.mgz "
viewcommand += "input/ernie/ernie_brain.mgz "
viewcommand += rigid + "abbytoernie.mgz "
viewcommand += affine + "abbytoernie_affine.mgz "



axis, idx = 1, 100

abby = nibabel.load("input/abby/abby_brain.mgz").get_fdata()
ernie = nibabel.load("input/ernie/ernie_brain.mgz").get_fdata()
rimg = nibabel.load(rigid + "abbytoernie.mgz").get_fdata()
aimg = nibabel.load(affine + "abbytoernie_affine.mgz").get_fdata()

fig, ax = plt.subplots(1,3)

ax[0].hist(abby.flatten())
ax[1].hist(ernie.flatten())
ax[2].hist(np.abs(abby-ernie).flatten())

for a in ax:
    a.set_yscale("log")
    a.set_xlim(0, 255)

ax[0].set_title("Abby")
ax[1].set_title("Ernie")
ax[2].set_title("Abby-Ernie")

# breakpoint()

fig, ax = plt.subplots(2,2)
# ax = ax.tolist()
ax = ax.flatten()
#breakpoint()

names = ["ernie", "abby", "abby2ernie", "abbytoernie_affine"]
images= [ernie, abby, rimg, aimg]

for a, name, img in zip(ax, names, images):
    # print(a, name, img)
    # breakpoint()
    # print(type(a), type(name), type(img))
    a.imshow(np.take(img, idx, axis), cmap="Greys_r")

    a.set_title(name)

fig, ax = plt.subplots(1,4)
# ax[0].imshow(np.abs(np.take(ernie-rimg, idx, axis)), cmap="jet", vmin=0, vmax=100)
# ax[1].imshow(np.abs(np.take(ernie-aimg, idx, axis)), cmap="jet", vmin=0, vmax=100)

def sigma(residual, mean=False):

    if mean:
        return sigma3(residual=residual)
    else:

        return 1.486 * np.median(np.abs(residual[np.abs(residual)>0] - np.median(residual[np.abs(residual)>0])))

# def sigma2(residual):
#     return 1.486 * np.mean(np.abs(residual[np.abs(residual)>0] - np.mean(residual[np.abs(residual)>0])))

def sigma3(residual):
    return np.mean((residual - np.mean(residual))**2)

def residual(x,y):
    # s = np.mean(y) / np.mean(x)
    
    s = 1
    print("s=", s)

    return x*np.sqrt(s) - y / np.sqrt(s)

f = 1

def l2loss(x, y, mean=False):
    r = residual(x, y)
    
    if f is None:
        factor =  sigma(r, mean)
    else:
        factor = f

    print("factor", factor)
    return (r / factor)**2


def tukeymean(x, y, c=4, mean=False):
    r = residual(x, y)
    if f is None:
        factor =  sigma(r, mean)
    else:
        factor = f

    print("factor", factor)

    return tukey2(r / factor, c=c)

vmax = None




abby = (aimg - np.mean(aimg)) / (np.std(aimg)**2)
ernie = (ernie - np.mean(ernie)) / (np.std(ernie)**2)

cc = 0.2

pcm1=ax[0].imshow(np.take(l2loss(abby, ernie), idx, axis), cmap="jet", vmax=vmax)
pcm2=ax[1].imshow(np.take(l2loss(abby, ernie, mean=True), idx, axis), cmap="jet", vmax=vmax)
pcm3=ax[2].imshow(np.take(tukeymean(abby, ernie, c=cc), idx, axis), cmap="jet", vmax=vmax)
pcm4=ax[3].imshow(np.take(tukeymean(abby, ernie, c=cc, mean=True), idx, axis), cmap="jet", vmax=vmax)

# ax[0].set_title(r"$(\sqrt{s} \frac{x}{\overline{x}} - \frac{1}{\sqrt{s}} \frac{y}{\overline{y}} )^2$")
# ax[1].set_title(r"tukey$(\sqrt{s} \frac{x}{\overline{x}}-\frac{1}{\sqrt{s}} \frac{y}{\overline{y}}, c=4.0)$")
# plt.suptitle("$s = \overline{y} / \overline{x}, \overline{z}$ = geometric mean")

ax[0].set_title(r"$(r/\sigma)^2$")
ax[1].set_title(r"$(r/\tilde\sigma)^2$")
ax[2].set_title(r"tukey$(r/\sigma, c=4.0)$")
ax[3].set_title(r"tukey$(r/\tilde\sigma, c=4.0)$")
plt.suptitle(r"x,y: Images, $s = \overline{y} / \overline{x}, \overline{x}$ = geometric mean of x" + "\n" + r"$r = \sqrt{s}x-\frac{y}{\sqrt{s}}$")

fig.colorbar(pcm1, ax=ax[0])
fig.colorbar(pcm2, ax=ax[1])
fig.colorbar(pcm3, ax=ax[2])
fig.colorbar(pcm4, ax=ax[3])
images = []

for image in ["cropped_abby_brain.mgz", "cropped_ernie_brain.mgz", "cropped_abbytoernie_affine.mgz"]:
    imagepath = affine + image

    image = nibabel.load(imagepath).get_fdata()

    images.append(image)


# def op(x,y):
#     return x-y

# for s in [sigma, sigma2, sigma3]:

#     print(sigma(op(images[0],images[1])))
#     print(sigma(op(images[0],images[2])))
#     print(sigma(op(images[2],images[1])))


plt.show()




print(viewcommand)