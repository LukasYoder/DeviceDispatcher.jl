# DeviceDispatcher.jl
*A lightweight dispatcher for **heterogeneous compute** in Julia*


![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)
[![DOI](https://zenodo.org/badge/1019761953.svg)](https://doi.org/10.5281/zenodo.15889537)


DeviceDispatcher.jl offers one-line / one-macro switching between the CPU
and **any GPU backend that is available** in the current Julia
environment.

It is purposely minimal—no macros that rewrite kernels, no exotic array
wrappers—yet it unlocks a productive workflow for laptops, workstations,
and clusters where *some* users have NVIDIA cards, *others* have AMD or
Apple Silicon, and *everyone* still wants the same source code to run.


---


## Key Features

| Feature                                      | Notes                                                                                                           |
|:---------------------------------------------|:----------------------------------------------------------------------------------------------------------------|
| **Auto‑Detect** CUDA, AMDGPU, Metal & oneAPI | loaded through Julia's package extension mechanism, so there is **zero hard dependency** on vendor packages     |
| **Seamless Data Movement**                   | `dev(x)` moves `x` to the active device (CPU <-> GPU) and works recursively on nested containers via `Adapt.jl` |
| **One‑Macro Context Switch**                 | `@gpu ... end` / `@cpu ... end` temporarily change the device on the calling task                               |
| **Thread‑Local**                             | each Julia thread keeps its own device flag; multithreaded programs can mix CPU & GPU work safely               |
| **Tiny Footprint**                           | one file, < 200 LOC, no extra allocations when the GPU path is inactive                                         |


---


## Installation

```julia
pkg> add DeviceDispatcher
```

Nothing else is required.

If a GPU package **is** present in the current environment,
DeviceDispatcher.jl automatically hooks into it:

* `CUDA.jl -> CuArray`
* `AMDGPU.jl -> ROCArray`
* `Metal.jl -> MtlArray`
* `oneAPI.jl -> OneArray`


---


## Quick Start

```julia
using DeviceDispatcher           # exports dev, @gpu, @cpu ...
using LinearAlgebra, Random

A = rand(Float32, 4096, 4096)
B = rand(Float32, 4096, 4096)

# ---------- CPU (default) -------------------------------------------
@time C = A * B                  # BLAS on the host

# ---------- Single GPU Section --------------------------------------
@gpu begin
    A_d = dev(A)                 # host -> device
    B_d = dev(B)
    @time C_d = A_d * B_d        # cuBLAS / rocBLAS / MPS ...
end                              # back to CPU automatically

# ---------- Persistent Mode -----------------------------------------
use_gpu!()                       # stay on the GPU
x = dev(rand(Float32, 10^7))
@time y = @. exp(x)              # broadcasted CUDA kernel
use_cpu!()                       # back to the host
```


---


## API in 20 Seconds

| Function / macro            | Purpose                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `dev(x)`                    | Convert `x` **to** the current device (CPU -> GPU or GPU -> CPU). Cascades through tuples, named tuples, arrays of arrays, etc. |
| `current_device()`          | `:cpu` or `:gpu` for the calling thread.                                                                                        |
| `use_gpu!()` / `use_cpu!()` | Permanently flip the thread‑local device flag.                                                                                  |
| `use\_device!(:cpu\|:gpu)`  | Convenience wrapper.                                                                                                            |
| `@gpu ...`, `@cpu ...`      | Temporarily run a block on GPU / CPU and restore the previous state on exit (even on error).                                    |


---


## How does it work?

* Each vendor backend lives in an **extension file**
  (`ext/DeviceDispatcherCUDAExt.jl`, *etc.*) that is *only* loaded when
  the user’s environment actually contains that package.
* The extension calls `_register_backend!(CUDA, CuArray)` (or the AMD /
  Metal / oneAPI equivalent).
  This installs a one‑liner closure that later, for example, turns
  `Array` -> `CuArray` via `Adapt.adapt`.
* Everything else is string‑free: no reflection, no `eval`, no needing
  to know the concrete array type in advance.  If a new backend lands
  tomorrow you can add an 8‑line extension and you’re done.


---


## Why not use GPUArrays.jl directly?

GPUArrays is fantastic for generic array code, but:

* It is *eager*—the constructor immediately copies to the first GPU it
  can find, even if the user was perfectly happy to stay on the CPU.
* It is **NVIDIA-centric** in practice; Apple/AMD users still need
  `Adapt` to move data to their native array type.
* It does not provide the high‑level "switch context" semantics that are
  convenient in ad‑hoc exploration or scripting.

DeviceDispatcher.jl is a lightweight, dependency‑free shim that fills exactly those
gaps.


---


## Heterogeneous Compute in Larger Projects

DeviceDispatcher.jl does *not* attempt to schedule work or hide data
transfers; its job is to make **data movement explicit, easy and safe**.
For full scale heterogeneous computing you can combine it with

* [`CUDA.jl`/`AMDGPU.jl`/`Metal.jl`/`oneAPI.jl`](https://juliagpu.org/)
  for custom kernels
* [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl)
  for portable kernel authoring
* [`Dagger.jl`](https://github.com/JuliaParallel/Dagger.jl) or
  [`Polyester.jl`](https://github.com/JuliaLinearAlgebra/Polyester.jl)
  for task‑based or BLAS‑like multi‑device scheduling.

DeviceDispatcher.jl happily coexists with all of them; it simply removes
the boilerplate of "wait, is this array on the right device?"


---


## License & Citation

**Copyright © 2025 Lukas Yoder**
Released under the [MIT License](LICENSE.md).

If you use *DeviceDispatcher.jl* in academic work, the following citation in your
acknowledgements section is appreciated:

> **Lukas Yoder**, "*LukasYoder/DeviceDispatcher.jl: Initial Release*". Zenodo, Jul. 15, 2025.
> DOI: [10.5281/zenodo.15889538](https://doi.org/10.5281/zenodo.15889538).


---

*Happy heterogeneous computing!* 🚀
