# Copyright (c) Zhihao Liang. All rights reserved.
import os
import torch
import numpy as np

import cumcubes

DENSITY_GRID = np.load(os.path.join(os.path.dirname(__file__), "data", "bunny.npy"))
print(f"DENSITY_GRID shape: ({DENSITY_GRID.shape[0]}, {DENSITY_GRID.shape[1]}, {DENSITY_GRID.shape[2]})")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    density_grid_cu = torch.tensor(DENSITY_GRID, device=device)
    with cumcubes.Timer("cuda marching cube: {:.6f}s"):
        vertices_cu, faces_cu = cumcubes.marching_cubes(density_grid_cu, 0, verbose=True) # verbose to print the number of vertices and faces
    with cumcubes.Timer("cumcubes save mesh: {:.6f}s\n"):
        cumcubes.save_mesh(vertices_cu, faces_cu, filename="bunny.ply")
