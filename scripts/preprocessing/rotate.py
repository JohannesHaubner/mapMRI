from scipy.spatial.transform import Rotation as R
import numpy as np

import nibabel

r = R.from_euler('zyx', [90, 45, 30], degrees=True)

r = r.as_matrix()

affine = np.zeros((4,4))


affine[:3, :3] = R

