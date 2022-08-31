from dolfin import *
parameters['ghost_mode'] = 'shared_facet'
#parameters["ghost_mode"] = "shared_vertex"

from PIL import Image
import numpy as np

def Pic2Fenics(FName):
    img = Image.open(FName)
    xPixel = np.shape(img)[0]
    yPixel = np.shape(img)[1]
    
    #which of the color channels to process
    Channels = (0,1,2)
    img.convert("RGB")
    
    mesh = RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(img.size[0], img.size[1]), int(img.size[0]), int(img.size[1]), "right")
    
    #Key mapping between global vertex index (input) and (i,j) pixel coordinate (output)
    #needs to be changed if the diagonal is not "right"
    PixID = np.zeros([2*img.size[0]*img.size[1], 2], dtype="uint")
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            #print "init:", i, j
            PixID[2*(img.size[0]*j + i)+0, 0] = img.size[1] - 1 - j
            PixID[2*(img.size[0]*j + i)+1, 0] = img.size[1] - 1 - j
            PixID[2*(img.size[0]*j + i)+0, 1] = i
            PixID[2*(img.size[0]*j + i)+1, 1] = i

    ImgSpace = VectorFunctionSpace(mesh, "DG", 0, len(Channels))
    ImgFunction = Function(ImgSpace)
    ImgFunction.rename("image", "")
    Fvalues = np.zeros(ImgFunction.vector().local_size())
        
    for chan in Channels:
        ImgDofs = ImgSpace.sub(chan).dofmap()
        cData = np.array(img.getchannel(chan))
        for c in cells(mesh):
            #local and global index of cell
            LID = c.index()
            GID = c.global_index()
            #local dof in DG0 function
            FID = ImgDofs.cell_dofs(c.index())[0]
            #get grey value from image
            MyGrey = cData[PixID[GID, 0], PixID[GID, 1]]
            #map 0..255 grey steps to [0,1]
            fValue = MyGrey/(255.0)
            Fvalues[FID] = fValue

    #Set function values
    ImgFunction.vector().set_local(Fvalues)
    ImgFunction.vector().apply("")

    return mesh, ImgFunction, len(Channels)

if __name__ == "__main__":
    InputFolder = "./"
    FName = "shuttle_small.png"
    (mesh, Img, NumData) = Pic2Fenics(FName)

    f = HDF5File(MPI.comm_world, FName+".h5", 'w')
    f.write(mesh, "mesh")
    #f.write(mf, "facet_region")
    f.write(Img, "Data1")
    f.close()
