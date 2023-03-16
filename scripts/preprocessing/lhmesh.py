import SVMTK as svmtk

import meshio

## NOTE lh.pial.stl can be created from lh.pial using mris_convert
# NOTE
stlfile = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/lh_mesh/lh.pial.stl"

fixedstlfile = stlfile.replace(".stl", "_fixed.stl")

surface = svmtk.Surface(stlfile)

# Remesh surface
surface.isotropic_remeshing(1, 3, False)

surface.smooth_taubin(5)

surface.fill_holes()

# Separate narrow gaps
# Default argument is -0.33. 
surface.separate_narrow_gaps(-0.33)


surface.save(fixedstlfile)


mm = meshio.read(fixedstlfile)
mm.write(fixedstlfile.replace(".stl", ".xml"))
mm.write(fixedstlfile.replace(".stl", ".xdmf"))