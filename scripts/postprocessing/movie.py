import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import nibabel


fig, ax = plt.subplots()


# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame

path = "/home/bastian/D1/registration/ventricle-outputs/446036/RKA0.01LBFGS50/states/"

# moviepath = path.replace("states/", "movies/")
moviepath = path
os.makedirs(moviepath, exist_ok=True)

files = sorted([path + x for x in os.listdir(path) if x.endswith(".mgz")])


for idx, axis in zip([125, 125, 125],[0, 1, 2]):

    images = []

    for i, img in enumerate(files):
        print(i, img)
        x = nibabel.load(img).get_fdata()

        images.append(np.take(x, idx, axis))
        
    ims = []

    for l in range(100):
        images.append(images[0] * (100 - l))

    for i, img in enumerate(images):

        plt.imshow(img, cmap="Greys_r", vmax=100)
        im = ax.imshow(img, animated=True)
        if i == 0:
            ax.imshow(img, cmap="Greys_r", vmax=100)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', length=0, width=0)
        ims.append([im])
        

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    # To save the animation, use e.g.
    #
    ani.save(moviepath + "idx" + str(axis) + "sl" + str(idx) + ".mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)