import torch
from torch import nn
from torch import fx
from typing import Dict, Any, Tuple
from .HRBatchNorm import HRBatchNorm2D
from .HRReLU import HRReLU
from .HRLinear import HRLinear

__all__ = ['change']


def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert (isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


@torch.no_grad()
def change(model: nn.Module) -> fx.GraphModule:
    traced_model = fx.symbolic_trace(model)
    named_modules = dict(traced_model.named_modules())
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            now_module = named_modules[node.target]
            if isinstance(now_module, nn.Linear):
                weight = now_module.weight
                bias = now_module.bias
                new_module = HRLinear(weight, bias)
                replace_node_module(node, named_modules, new_module)
            if isinstance(now_module, nn.ReLU):
                replace_node_module(node, named_modules, HRReLU())
            if isinstance(now_module, nn.BatchNorm2d):
                weight = now_module.weight
                bias = now_module.bias
                mean = now_module.running_mean
                std = torch.sqrt(now_module.running_var + now_module.eps)
                new_module = HRBatchNorm2D(mean, std, weight, bias)
                replace_node_module(node, named_modules, new_module)
    return traced_model
