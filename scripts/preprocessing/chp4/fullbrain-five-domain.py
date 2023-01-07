import SVMTK as svmtk
import argparse


def create_brain_mesh(stls, output, resolution, remove_ventricles=True):

    # Load each of the Surfaces
    surfaces = [svmtk.Surface(stl) for stl in stls]
    
    # Take the union of the left (#3) and right (#4)
    # white surface and put the result into
    # the (former left) white surface
    surfaces[2].union(surfaces[3])

    # ... and drop the right white surface from the list
    surfaces.pop(3)

    # Define identifying tags for the different regions 
    tags = {"pial": 1, "white": 2, "ventricle": 3}

    # Label the different regions
    smap = svmtk.SubdomainMap()
    smap.add("1000", tags["pial"]) 
    smap.add("0100", tags["pial"]) 
    smap.add("1010", tags["white"])
    smap.add("0110", tags["white"])
    smap.add("1110", tags["white"])
    smap.add("1011", tags["ventricle"])
    smap.add("0111", tags["ventricle"])
    smap.add("1111", tags["ventricle"])

    # Generate mesh at given resolution
    domain = svmtk.Domain(surfaces, smap)
    domain.create_mesh(resolution)

    # Remove ventricles perhaps
    if remove_ventricles:
        domain.remove_subdomain(tags["ventricle"])

    # Save mesh    
    domain.save(output)

stls = ("lh.pial.stl", "rh.pial.stl",
        "lh.white.stl", "rh.white.stl",
        "lh.ventricles.stl")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)

    args = vars(parser.parse_args())

    create_brain_mesh(stls, args["name"] + str(args["resolution"]) + ".mesh", resolution=args["resolution"])



