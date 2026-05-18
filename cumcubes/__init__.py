# Copyright (c) Zhihao Liang. All rights reserved.
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Callable

import torch
import numpy as np

from . import src as _C
from .utils import Timer, scale_to_bound


def marching_cubes(
    density_grid: torch.Tensor,
    thresh: float,
    scale: Optional[Union[float, Sequence]]=None,
    verbose: bool=False,
    device: Optional[Union[str, torch.device]]=None
) -> Tuple[torch.Tensor]:
    """
    density_grid: CUDA tensor [D, H, W], any floating dtype.
    return: vertices CUDA tensor [Nv, 3], faces CUDA tensor [Nf, 3].
    """
    assert torch.is_tensor(density_grid) and density_grid.is_cuda, "cumcubes.marching_cubes expects a CUDA tensor"
    assert density_grid.ndim == 3, density_grid.shape
    assert density_grid.shape[0] >= 2 and density_grid.shape[1] >= 2 and density_grid.shape[2] >= 2, density_grid.shape
    if device is not None:
        device = torch.device(device)
        assert device.type == "cuda" and device.index is not None, device
        assert density_grid.device == device, f"density_grid must be on {device}, got {density_grid.device}"

    lower: List[float]
    upper: List[float]
    if scale is None:
        lower = [0.0, 0.0, 0.0]
        upper = [density_grid.shape[0], density_grid.shape[1], density_grid.shape[2]]
    else:
        lower, upper = scale_to_bound(scale)

    density_grid = density_grid.to(torch.float32).contiguous()
    vertices, faces = _C.marching_cubes(density_grid, thresh, lower, upper)
    
    if verbose:
        print(f"#vertices={vertices.shape[0]}")
        print(f"#triangles={faces.shape[0]}")

    return vertices, faces


def marching_cubes_func(
    scale: Optional[Union[float, Sequence]]=None,
    num_x: int=100,
    num_y: int=100,
    num_z: int=100,
    func: Callable=lambda x, y, z: x**2 + y**2 + z**2,
    thresh: float=16,
    verbose: bool=False,
    device: Optional[Union[str, torch.device]]=None
) -> Tuple[torch.Tensor]:
    """
    device: explicit CUDA device for evaluating func and running marching cubes.
    return: vertices CUDA tensor [Nv, 3], faces CUDA tensor [Nf, 3].
    """
    assert device is not None, "cumcubes.marching_cubes_func expects an explicit CUDA device"
    device = torch.device(device)
    assert device.type == "cuda" and device.index is not None, device

    lower, upper = scale_to_bound(scale)
    x = torch.linspace(lower[0], upper[0], num_x)
    y = torch.linspace(lower[0], upper[0], num_y)
    z = torch.linspace(lower[0], upper[0], num_z)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    sample_points = torch.stack((X, Y, Z), dim=-1).reshape(num_x, num_y, num_z, 3).contiguous()

    vertices, faces = _C.marching_cubes_func(sample_points, thresh, lower, upper, func, device.index)
    
    if verbose:
        print(f"#vertices={vertices.shape[0]}\n")
        print(f"#triangles={faces.shape[0]}\n")

    return vertices, faces


def save_mesh(
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]]=None,
    filename: Union[str, Path]="temp.ply",
    verbose: bool=False
) -> None:
    """save mesh into the given filename

    Args:
        vertices (Union[torch.Tensor, np.ndarray]): vertices of the mesh to save
        faces (Union[torch.Tensor, np.ndarray]): faces of the mesh to save
        colors (Optional[Union[torch.Tensor, np.ndarray]], optional):
            vertices of the mesh to save. Defaults to None.
        filename (Union[str, Path], optional):
            the save path. Defaults to "temp.ply".
        verbose (bool, optional):
            print verbose mention or not. Defaults to False.
    """

    if isinstance(filename, Path):
        filename = str(filename)

    if isinstance(vertices, np.ndarray): vertices = torch.tensor(vertices)
    if isinstance(faces, np.ndarray): faces = torch.tensor(faces)

    # process colors
    if colors is None:
        colors = torch.ones_like(vertices) * 127
    elif isinstance(colors, np.ndarray):
        colors = torch.tensor(colors)
    colors = colors.to(torch.uint8)

    if filename.endswith(".ply"):
        _C.save_mesh_as_ply(filename, vertices, faces, colors)
    else:
        raise NotImplementedError()
    
    if verbose:
        print(f"save as {filename} successfully!")


__all__ = ["Timer", "marching_cubes", "save_mesh"]
