import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def save_tripcolor(
        mesh,
        perm,
        save_path,
        title=None
):

    """
    Draw FEM conductivity distribution

    mesh:
        PyEITMesh

    perm:
        (1422,)
        conductivity on elements
    """


    nodes = mesh.node

    elements = mesh.element


    x = nodes[:,0]

    y = nodes[:,1]


    triangulation = tri.Triangulation(
        x,
        y,
        elements
    )


    plt.figure(
        figsize=(5,5)
    )


    plt.tripcolor(
        triangulation,
        perm,
        shading="flat"
    )


    plt.colorbar()


    if title is not None:

        plt.title(title)


    plt.axis(
        "equal"
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300
    )


    plt.close()