# Copyright (c) Zhihao Liang. All rights reserved.
import torch

import cumcubes

# Create a data volume (30 x 30 x 30)
def sphere_f(x: float, y: float, z: float) -> float:
    return x**2 + y**2 + z**2


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.randn(10, device=device) # allocate the GPU memory
    with cumcubes.Timer("cuda marching cube func: {:.6f}s"):
        vertices_cu, faces_cu = cumcubes.marching_cubes_func(
            ((-10, -10, -10), (10, 10, 10)),    # Bounds
            100, 100, 100,                      # Number of samples in each dimension
            sphere_f,                           # Implicit function
            16,                                 # Thresh
            device=device
        )
    with cumcubes.Timer("cumcubes save mesh: {:.6f}s\n"):
        cumcubes.save_mesh(vertices_cu, faces_cu, filename="sphere_func.ply")
