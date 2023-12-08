#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]

__all__ = ["meshgrid"]


def meshgrid(*tensors):
    """
    生成网格, tensor可以是2个及以上的1维tensor.
    比如x长度为3, y长度为4, 
        grid_x, grid_y = torch.meshgrid(x, y)
    grid_x为将一个x作为一行, 复制4次; grid_y为将y在列方向上复制3次:
        print(grid_x)
        # tensor([[1, 2, 3],
        #         [1, 2, 3],
        #         [1, 2, 3],
        #         [1, 2, 3]])
        print(grid_y)
        # tensor([[4, 4, 4],
        #         [5, 5, 5],
        #         [6, 6, 6],
        #         [7, 7, 7]])

    """
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)
