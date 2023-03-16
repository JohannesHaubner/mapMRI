import SVMTK as svmtk
import meshio
import os

surfacemesh = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/transported-meshes/"
surfacemesh += "abby-brain/dgmeshtransform-4/"
surfacemesh += "transformed_regaff_boundary.xml"

assert "boundary" in surfacemesh

inputfile = surfacemesh.replace(".xml", ".stl")
outputname = surfacemesh.replace(".xml", "_postprocessed.stl")
outputname2 = surfacemesh.replace(".xml", "_postprocessed2.stl")

os.system("rm -v " + outputname2)

# os.system("meshio-convert " + surfacemesh + " " + inputfile)

# assert os.path.isfile(inputfile)

# surface = svmtk.Surface(inputfile)

# surface.fill_holes()

# surface.smooth_taubin(5)

# # surface.isotropic_remeshing(1, 3, False)



# print(outputname)
# surface.save(outputname)

assert os.path.isfile(outputname)

Lines = open(outputname).readlines()

for idx, line in enumerate(Lines):
    elements = line.split(" ")

    if len(elements) != 1:

        make_line = False
        
        for e in elements:
            
            if "vertex" in e or "facet" in e or "normal" in e:
                continue
            if "loop" in e or "outer" in e:
                continue
            else:
                if (e.count("e")) > 0:
                    make_line = True
        
        if make_line:
            newline = ""
            for e in elements:
                try:
                    
                    ee = e.replace("\n", "")
                    float(ee)
                    newline += format(round(float(ee), 5), ".5f")

                except ValueError:
                    if not ("facet" == e or "normal" == e):
                        print("---------", e)
                        raise ValueError()
                    
                    newline += e
                if idx +1 != len(elements):
                    newline += " "

            print(line.replace("\n", ""))
            print("-->", newline.replace("\n", ""))
            line = newline


    if idx == 0:

        text_file = open(outputname2, "w")
        text_file.write(line)
        if not line.endswith("\n"):
            text_file.write("\n")

    else:
        text_file = open(outputname2, "a")

        text_file.write(line)
        if not line.endswith("\n"):
            text_file.write("\n")
    
text_file.close()

outputname = outputname2

print("Created", outputname, "trying to convert with meshio")

os.system("meshio-convert " + outputname + " " + outputname.replace(".stl", ".mesh"))

assert os.path.isfile(outputname.replace(".stl", ".xml"))

os.system("meshio-convert " + outputname.replace(".stl", ".mesh") + " " + outputname.replace(".stl", ".xdmf"))