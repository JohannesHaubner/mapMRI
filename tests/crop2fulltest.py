from dgregister.meshtransform import crop_to_original
import nibabel
from dgregister.helpers import view
import numpy as np
import os

# crop_file = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/ventricles/021to068_nyul_ventriclemasked.mgz"
crop_file = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/ventricle-outputs/445351/RKA0.01LBFGS50/CurrentState.mgz"

original = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/normalized/registered/021to068.mgz"

box = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/freesurfer/021/testouts/box_all.npy"

# TODO FIXME move stuff from testouts to proper place

comd = "freeview "
comd += crop_file + " "
comd += original + " "
comd += "back2orig.mgz "
os.system(comd)

crop = nibabel.load(crop_file).get_fdata()

box = np.load(box)

orig_image = nibabel.load(original)

croptoorig = crop_to_original(orig_image=orig_image, cropped_image=crop, box=box, space=2, pad=2)

orig = nibabel.load(original)
orig_affine = orig.affine
orig = orig.get_fdata()

# dimg = np.abs(np.where(croptoorig > 0, 100, 0)-np.where(orig > 0, 100, 0))
# dimg = np.abs(croptoorig-orig)
# dimg = np.where(dimg > 5, 50, 0)

view([# crop, 
    croptoorig, orig, # dimg
      ], axis=1, idx=100)


nibabel.save(nibabel.Nifti1Image(croptoorig, orig_affine), "back2orig.mgz")

comd = "freeview "
comd += crop_file + " "
comd += original + " "
comd += "back2orig.mgz "
os.system(comd)