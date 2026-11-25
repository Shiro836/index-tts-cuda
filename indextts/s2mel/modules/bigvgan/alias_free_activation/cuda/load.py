# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import importlib.util
import os
import pathlib
import subprocess
import sys

from torch.utils import cpp_extension

"""
Setting this param to a list has a problem of generating different compilation commands (with diferent order of architectures) and leading to recompilation of fused kernels. 
Set it to empty stringo avoid recompilation and assign arch flags explicity in extra_cuda_cflags below
"""
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load():
    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
    if int(bare_metal_major) >= 13:
        # CUDA 13 dropped compute_70; sm_80 cubins do not run on Blackwell.
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_120,code=sm_120")

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    prebuilt = _load_prebuilt(buildpath)
    if prebuilt is not None:
        return prebuilt

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=[
                "-O3",
                # compute_70 (Volta) was dropped in CUDA 13; nvcc fatals on it.
                "--use_fast_math",
            ]
            + extra_cuda_flags
            + cc_flag,
            verbose=True,
        )

    extra_cuda_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    sources = [
        srcpath / "anti_alias_activation.cpp",
        srcpath / "anti_alias_activation_cuda.cu",
    ]
    anti_alias_activation_cuda = _cpp_extention_load_helper(
        "anti_alias_activation_cuda", sources, extra_cuda_flags
    )

    return anti_alias_activation_cuda


def _load_prebuilt(buildpath):
    """Load an already-compiled anti_alias_activation_cuda.so, skipping ninja.

    g++-15 on this host cannot compile the installed torch headers (GCC 15
    promotes -Wtemplate-body to a hard error). The sibling indextts/BigVGAN
    build ships a working sm_80 cubin, which also runs on Ada (sm_89). Set
    INDEXTTS_BIGVGAN_PREBUILT_SO to override the path.
    """
    override = os.environ.get("INDEXTTS_BIGVGAN_PREBUILT_SO")
    so = pathlib.Path(override) if override else pathlib.Path(buildpath) / "anti_alias_activation_cuda.so"
    if not so.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("anti_alias_activation_cuda", so)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as ex:
        print(f"prebuilt BigVGAN kernel at {so} did not load ({ex}); rebuilding")
        return None
    if not hasattr(module, "forward"):
        print(f"prebuilt BigVGAN kernel at {so} has no forward(); rebuilding")
        return None
    sys.modules["anti_alias_activation_cuda"] = module
    return module


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
