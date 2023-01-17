import nibabel
import numpy as np
import os
import SVMTK as svm
from fenics import *

def meshinfo(meshfile):
    m = Mesh(meshfile)

    boundarymesh = BoundaryMesh(m, "exterior")
    nvertices = m.num_vertices()
    nboundaryvertices = boundarymesh.num_vertices()

    print(meshfile)
    print("Vertices =", format(nvertices, ".2e"))

    print("Boundary vertices =", format(nboundaryvertices, ".2e"))
    print("That means interior vertices =", nvertices- nboundaryvertices)
    print("Ratio mesh.num_vertices() / boundarymesh.num_vertices() =", nvertices / nboundaryvertices)


    V = FunctionSpace(m, "CG", 1)
    f = interpolate(Constant(1), V)
    vol = assemble(f*dx)

    print("Volume =", format(vol, ".2e"), "mm ** 3")

    print("Vertices / mm ** 3 =", format(nvertices / vol, ".2e"))

    print("---------------------------------------------")

    return nvertices

    
overwrite = False
print("overwrite", overwrite)
breakpoint()


# Crude workaround due to incompatibility between python and bash directories
# assert os.getcwd() == "/home/basti/Dropbox (UiO)/mripinn"

os.chdir("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d")

path = "./"

roiname = "newtarget"
roiname = "input"
meshpath = path
mask_to_be_meshed = meshpath + roiname + ".mgz"


assert "Dropbox" not in mask_to_be_meshed

os.makedirs(meshpath, exist_ok=True)
os.system("ls " + meshpath)

res = 8

threshold = 0.05
meshfile = meshpath + roiname + "%d.mesh" % res

assert os.path.isfile(mask_to_be_meshed)

surffile = meshpath + roiname + "meshsurface.stl"

if not os.path.isfile(surffile) or overwrite:

    cmd2 = 'mri_volcluster --in ' + mask_to_be_meshed + ' --thmin ' + str(threshold) + '  --minsize 10 --ocn tmp.mgz'
    os.system(cmd2)

    cmd3 = 'mri_binarize --i tmp.mgz --match 1 --o tmp2.mgz'
    os.system(cmd3)

    cmd4 = 'mri_morphology tmp2.mgz close 2 tmp2.mgz'

    cmd5 = 'mri_binarize --i tmp2.mgz --match 1 --surf-smooth 1 --surf ' + meshpath + roiname + 'meshsurface.stl'
    os.system(cmd5)
    
    cmd5 = 'mri_binarize --i tmp2.mgz --match 1 --surf-smooth 1 --surf ' + meshpath + roiname + 'meshsurface'
    os.system(cmd5)

    # os.system('rm tmp.lut; rm tmp.mgz; rm tmp2.mgz')
print("*"*80)
print("freeview " + roiname+ ".mgz tmp.mgz -f " + roiname + 'meshsurface')
# exit()

xmlfile = meshpath + roiname + "%d.xml" % res

if not os.path.isfile(meshfile) or overwrite:
    surf = svm.Surface(surffile)
    domain = svm.Domain([surf])

    domain.create_mesh(res)
    domain.save(meshfile)

    
    os.system("meshio-convert " + meshfile + " " + meshpath + roiname + "%d.xdmf" % res)
    os.system("meshio-convert " + meshfile + " " + xmlfile)


nvertices = meshinfo(xmlfile)


exit()

if not (nvertices >= 1.e5 and nvertices <= 1.12e5):
    raise ValueError()

markerfile = meshpath + "mesh-markerfun" + roiname + str(res) + ".hdf"

if not os.path.isfile(markerfile) or overwrite:
    from dolfin import interpolate, Expression, HDF5File, Mesh, FunctionSpace, File
    mesh = Mesh(meshpath + roiname + "%d.xml" % res)

    print("mesh hmax from dolfin:", mesh.hmax())

    V = FunctionSpace(mesh, "CG", 2)
    markerfun = interpolate(Expression("1", degree=0), V)

    pvd = File(meshpath + roiname + "%d.pvd" % res)
    pvd << markerfun

    Hdf = HDF5File(mesh.mpi_comm(), markerfile, "w")
    Hdf.write(mesh, "mesh")
    Hdf.write(markerfun, "markerfun")
    Hdf.close()

meshmask = meshpath + "meshmask" + str(res)

if not os.path.isfile(meshmask + ".mgz") or overwrite:
    # import torchpinns
    from torchpinns.mri.simdata_utils import make_mask 
    
    make_mask(
        meshfile=meshfile[:-4] + "xml",
        imagefile=mask_to_be_meshed, # "./wmparc_edited.mgz",
        markerfun=markerfile,
        outputname=meshmask,
        output_mgz=True, exportfolder="",
        scriptpath="/home/basti/programming/brainsim/scripts/fenics_function_to_image.py"
    )
