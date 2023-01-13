
import SVMTK as svm


if __name__ == "__main__":

   # path = "../FEniCS/parenchyma/diffusion_simulation_code/simulation_inputs/parenchyma_mesh.h5"

   # path = "../FEniCS/parenchyma/xmlmesh.xml"

   # path = "./meshmesh.mesh"

   path = "../../Documents/MRI/PatID-068/surf/lh.pial.stl"

   # svm.Domain(path)

   surf = svm.Surface(path)

   # surf.fill_holes()

   #surf = surf.convex_hull()

   

   # surf.smooth_laplacian(0.5, 20)

   surf.separate_narrow_gaps(-0.3)

   surf.smooth_taubin(10)

   # surf.isotropic_remeshing(1, 3, True)

   slc = surf.slice(1, 0, 0, 28.)
   #slc = surf.slice(0, 1, 0, 9.)

   # breakpoint()

   # breakpoint()
   
   # slc.simplify(100.)
   slc.create_mesh(16)
   # slc.keep_largest_connected_component()
   # slc.simplify(.5)
   # breakpoint()
   # slc.save("./test.vtu")
   slc.save("./test.mesh")

   # Can call this after create_mesh:
   # slice_surface = slc.as_surface()
   # slice_surface.fill_holes()   

   # # slice_surface.keep_largest_connected_component()

   # domain = svm.Domain(slice_surface)
   # domain.create_mesh(64)
   # domain.save("./cc.mesh")



