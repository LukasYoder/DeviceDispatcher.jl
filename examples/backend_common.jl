#!/usr/bin/env julia

using DeviceDispatcher
using DeviceDispatcher: dev, use_gpu!, use_cpu!, use_device!, current_device, @gpu, @cpu

using Logging
using LinearAlgebra, Random


println("---------------------- DeviceDispatcher Self-Test ----------------------")
println("Detected backend : ",
        isnothing(DeviceDispatcher._backend_mod) ?
          "none (CPU only)" : DeviceDispatcher._backend_mod)


# ----------------------------------------------------------------
# CPU Reference Computation
# ----------------------------------------------------------------
Random.seed!(1)
A = rand(Float32, 2048, 2048)
B = rand(Float32, 2048, 2048)

println()
println("CPU Benchmark (2048×2048 GEMM)")
@time C_cpu = A * B


# ----------------------------------------------------------------
# GPU Benchmark Using the `@gpu` Macro
# ----------------------------------------------------------------
try
  println()
  println("GPU Benchmark with `@gpu` Macro")
  @gpu begin
    A_d = dev(A)
    B_d = dev(B)
    @time C_gpu = A_d * B_d
  end
catch err
  println("GPU run skipped: ", err.msg)
end


# ----------------------------------------------------------------
# Persistent Switch Example
# ----------------------------------------------------------------
try
  println()
  println("Persistent GPU Mode")
  use_gpu!()
  x = dev(rand(Float32, 10^6))
  @time y = @. log1p(x)
  use_cpu!()
catch err
  println("Persistent GPU demo skipped: ", err.msg)
end


println("------------------ DeviceDispatcher Self-Test Complete -----------------")
