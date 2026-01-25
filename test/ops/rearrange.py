import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, random_int_tensor


def torch_rearrange(out, inp):
    # This is essentially a copy operation that respects strides
    # Logic: make sure out accepts data from inp.
    # In PyTorch, we can just copy inp to out.
    out.copy_(inp)


def test_op_rearrange(
    shape,
    out_stride_order=None,
    in_stride_order=None,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="device",
    profile=False,
):
    # out_stride_order/in_stride_order: tuple of indices representing the permutation of strides
    # e.g for shape (2,3,4), default stride is (12, 4, 1).
    # if stride_order is (2, 0, 1), we permute the physical memory layout essentially?
    # Actually, easy way to test rearrange is to create non-contiguous tensors in torch/llaisys and copy between them.

    # 1. Create source tensor (possibly non-contiguous)
    gen_tensor = random_tensor
    if dtype_name == "i64":
        gen_tensor = random_int_tensor
        
    inp, inp_ = gen_tensor(shape, dtype_name, device_name)
    if in_stride_order:
        inp = inp.permute(in_stride_order)
        # For llaisys, we need to manually adjust strides or create view?
        # Llaisys `random_tensor` helper (implied) might not support creating strided tensors directly easily
        # except via permute/transpose which is usually what rearrange handles?
        # But wait, llaisys.Ops.rearrange(out, in) takes two existing tensors.
        # The prompt says llaisys rearrange implementation calls cpu::rearrange with shapes/strides.
        # So we need to construct llaisys tensors with specific strides.
        
        # Simpler approach: Create a tensor, permute it, and that becomes our input with custom strides.
        inp_ = inp_.permute(in_stride_order)
    
    # 2. Create destination tensor (possibly non-contiguous)
    out, out_ = gen_tensor(inp.shape, dtype_name, device_name)
    if out_stride_order:
         # To simulate a non-contiguous output buffer we want to write into:
         # Create a larger base tensor and slice/permute? 
         # Or just permute.
         # For testing, we want 'out' to have specific strides.
         # PyTorch: out = tensor.permute(...) returns a view.
         # If we write to this view, it writes to underlying storage in strided manner.
         base_out_shape = [shape[i] for i in out_stride_order] # This logic is tricky for pure permute.
         
         # Let's just stick to permute on both sides.
         # Example: Shape (2, 3). Permute(1, 0) -> (3, 2).
         # If input is (3, 2) (originally (2,3)), and output is (3, 2) (originally (2,3)),
         # we copy from one view to another.
         
         # Wait, if we change shape via permute, `check_equal` might compare potentially different shapes if not careful.
         # But the op checks SAME_SHAPE. So input and output must have same logical shape.
         # But they can have different strides.
         
         # Correct Setup:
         # logical shape: S
         # Input: Tensor of shape S, strides Str_in
         # Output: Tensor of shape S, strides Str_out
         
         # How to get specific strides in PyTorch/Llaisys test harness?
         # .permute() changes shape AND strides.
         # .transpose() changes shape AND strides.
         # To get same shape but different strides:
         # Create (3, 2) -> permute(1, 0) -> (2, 3) (strides swapped).
         # Create (2, 3) (strides normal).
         # Copy from (2, 3)[strided] to (2, 3)[contiguous] or vice versa.
         
         out = out.permute(out_stride_order)
         out_ = out_.permute(out_stride_order)
    
    print(f"   shape={inp.shape} stride_in={inp.stride()} stride_out={out.stride()} dtype <{dtype_name}>")

    torch_rearrange(out, inp)
    llaisys.Ops.rearrange(out_, inp_)
    
    assert check_equal(out_, out, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_rearrange(out, inp),
            lambda: llaisys.Ops.rearrange(out_, inp_),
            device_name,
        )

def run_permutation_test(base_shape, perm, dtype, atol, rtol, device_name):
    # In this case we create a tensor of 'base_shape', permute it to get 'perm_shape'.
    # We test copying FROM this permuted tensor TO a contiguous tensor of 'perm_shape'.
    # And vice versa.
    
    if dtype == "i64":
        # random_int_tensor(shape, device_name, dtype_name="i64", ...)
        gen_tensor = lambda s, d, dev: random_int_tensor(s, dev, d)
    else:
        # random_tensor(shape, dtype_name, device_name, ...)
        gen_tensor = random_tensor

    # 1. Non-Contig -> Contig
    inp, inp_ = gen_tensor(base_shape, dtype, device_name)
    # print(f"DEBUG: inp_.shape={inp_.shape()} ndim={inp_.ndim()} perm={perm}") 
    # Commented out print, just ensuring permute call is correct.
    # Note: llaisys.Tensor.shape() returns list/vector.
    
    inp = inp.permute(*perm)
    inp_ = inp_.permute(*perm) # Now shape is perm_shape, non-contiguous
    
    target_shape = inp.shape
    out, out_ = gen_tensor(target_shape, dtype, device_name) # Contiguous
    
    print(f"   [NC->C] shape={target_shape} dtype <{dtype}>")
    torch_rearrange(out, inp)
    llaisys.Ops.rearrange(out_, inp_)
    assert check_equal(out_, out, atol, rtol)

    # 2. Contig -> Non-Contig
    inp, inp_ = gen_tensor(target_shape, dtype, device_name) # Contiguous source
    
    out, out_ = gen_tensor(base_shape, dtype, device_name)
    out = out.permute(*perm)
    out_ = out_.permute(*perm) # Target is non-contiguous view
    
    print(f"   [C->NC] shape={target_shape} dtype <{dtype}>")
    torch_rearrange(out, inp)
    llaisys.Ops.rearrange(out_, inp_)
    assert check_equal(out_, out, atol, rtol)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    
    # Test cases
    # 1. Contiguous to Contiguous copy
    # 2. Transposed to Contiguous
    # 3. Contiguous to Transposed
    # 4. Transposed to Transposed
    
    # Shapes where permute makes sense (must be > 1D for interesting strides)
    test_metas = [
       # Shape, out_perm, in_perm
       ((10, 20), None, None), # C -> C
       ((10, 20), None, (1, 0)), # T -> C (Input is transposed view of (20, 10)) -> Wait, Logical shape becomes (20, 10).
       # If I want logical shape to be (10, 20), I should start with (20, 10) and transpose.
    ]
    
    testDtypePrec = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
        ("i64", 0, 0),
    ]

    print(f"Testing Ops.rearrange on {args.device}")

    # Case 1: Simple Copy (Continuous)
    for dtype_name, atol, rtol in testDtypePrec:
        if dtype_name == "i64":
             # We can't reuse test_op_rearrange cleanly because it uses random_tensor inside.
             # Let's just run run_permutation_test with Identity perm?
             # Or modify test_op_rearrange as well.
             # run_permutation_test((10, 30), (0, 1), dtype_name, atol, rtol)
             # perm (0, 1) is identity for 2D.
             # NOTE: random_tensor wrapper logic is now in run_permutation_test.
             run_permutation_test((10, 30), (0, 1), dtype_name, atol, rtol, args.device)
        else:
             test_op_rearrange((10, 30), None, None, dtype_name, atol, rtol, args.device, args.profile)
        
    # Case 2: Input Transposed (Non-contiguous read)
    # Logical shape (10, 20).
    # Input: Created as (20, 10), then transposed (1, 0) -> (10, 20) with strides (1, 10). 
    # Output: Created as (10, 20), strides (20, 1).
    for dtype_name, atol, rtol in testDtypePrec:
         # We need to pass the permutation required to REACH the target logical shape from the creation shape if we want to change strides.
         # But verify helper takes 'shape' as creation shape. 
         # Let's adjust logic in test_op_rearrange to be more explicit.
         pass
         
    # Redefine test runner for clarity
    
    # (2, 3, 4) -> permute(0, 2, 1) -> (2, 4, 3). 
    # Compare copying (2, 4, 3)[strided] to (2, 4, 3)[contig]

    
    perms = [
        ((10, 20), (1, 0)),
        ((4, 5, 6), (0, 2, 1)),
        ((4, 5, 6), (2, 1, 0)),
    ]
    
    for shape, perm in perms:
        for dtype, atol, rtol in testDtypePrec:
            run_permutation_test(shape, perm, dtype, atol, rtol, args.device)

    print("\033[92mTest passed!\033[0m\n")
