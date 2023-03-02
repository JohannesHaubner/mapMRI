import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import nibabel
from dgregister.helpers import get_bounding_box_limits



# path = "/home/bastian/D1/registration/ventricle-outputs/446036/RKA0.01LBFGS50/states/"
# path = "/home/bastian/D1/registration/ventricle-outputs/446058/RKA0.01LBFGS50/states/"
# box = "/home/bastian/D1/registration/hydrocephalus/freesurfer/021/testouts/box_all.npy"


path = "/home/bastian/D1/registration/normalized-outputs/446152/RKA0.01LBFGS100/states/"
box = "/home/bastian/D1/registration/mri2fem-dataset/normalized/cropped/box.npy"

# /home/bastian/D1/registration/normalized-outputs/446152/RKA0.01LBFGS100/states/

if "bastian" not in os.getcwd():
    b = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/ventricle-outputs/"

    path = path.replace("/home/bastian/D1", "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation")
    box = box.replace("/home/bastian/D1", "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation")


box = np.load(box)

# moviepath = path.replace("states/", "movies/")
moviepath = path
os.makedirs(moviepath, exist_ok=True)

files = sorted([x for x in os.listdir(path) if x.endswith(".mgz")], key=lambda x: int(x.replace("state", "").replace(".mgz", "")))
files = [path + x for x in files]

for idx, axis in zip([125, 125, 125],[0, 1, 2]):

    images = []

    limits = get_bounding_box_limits(np.take(box, idx, axis))

    print(limits)

    for i, img in enumerate(files[:]):
        print(i, img)
        x = nibabel.load(img).get_fdata()

        images.append(np.take(x, idx, axis))
        
    ims = []
    fig, ax = plt.subplots(ncols=3)

    if not isinstance(ax, np.ndarray):
        ax = [None, ax]

    for i, img in enumerate(images[:-1]):

        im = ax[1].imshow(img, cmap="Greys_r", vmin=0, vmax=100, animated=True)
        
        if i == 0:
            ax[1].imshow(img, cmap="Greys_r", vmin=0, vmax=100)

        ax[2].imshow(images[-1], cmap="Greys_r", vmin=0, vmax=100)
        ax[2].set_title("Target")
        
        ax[0].imshow(images[0], cmap="Greys_r", vmin=0, vmax=100)
        ax[0].set_title("Input")

        for a in ax:
            a.set_xticks([])
            a.set_xlim(limits[1].start, limits[1].stop)
            a.set_ylim(limits[0].stop, limits[0].start)
            a.set_yticks([])
            a.tick_params(axis='both', length=0, width=0)

        ims.append([im])
        
    plt.tight_layout()

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    # To save the animation, use e.g.
    #
    ani.save(moviepath + "idx" + str(axis) + "sl" + str(idx) + ".mp4")

    print("Done with animation for direction ", axis)
    # print("Exit !")
    # exit()

    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)