# pybind_v_dispatcher

Moving a PyTorch op to the stable ABI means registering it with the dispatcher, so leaving
`pybind11` is mandatory on that path — this measures what that step costs. The same C++
symbol is exposed three ways (`PYBIND11_MODULE`, `TORCH_LIBRARY`, `STABLE_TORCH_LIBRARY`),
so the binding is the only variable. Timing is `torch.utils.benchmark.blocked_autorange`
with `num_threads=1`; the arms are checked for identical results before anything is timed.
The toll is device-independent — measured at +473 ns on CPU and +476 ns on CUDA tensors,
a difference well inside the IQR — so it is measured on CPU and divided into a separately
measured op cost, rather than recovered by differencing two large timings.

What the toll is a fraction of depends entirely on the device. A CPU op's cost grows with
`numel`, so the toll decays to ~0.1%; a dispatched CUDA op is launch-bound at ~5.5 µs of
host time regardless of size, so the toll sits at a flat ~8% until the kernel is large
enough to backpressure the launch queue. That is why a decode-shaped GPU workload pays the
toll on every eager call, per layer per token, while a training-shaped one does not notice
it. These are eager-mode numbers; a captured CUDA graph replays launches without the
dispatcher and does not pay it at all.
It is also not pure overhead: a `pybind11` call fails `torch.compile(fullgraph=True)` and
emits no profiler event, so the dispatcher charges for work the pybind arm never does.
To measure your own op, replace the marked block in `bindings.cpp` (and `stable.cpp` for
arm C) and run it a few times — between-run drift exceeded the within-run IQR on our box.

## Run

```bash
python setup.py build_ext --inplace
python bench.py                 # or: python bench.py 5.0
```

```
torch 2.15.0a0+gitc312cf8 | blocked_autorange(min_run_time=2.0s) | num_threads=1
all arms return identical results

THE TOLL (identity body: pure call overhead)
  arm                              no grad      autograd   ns/call (IQR)
  A  pybind11                      94 (  2)       94 (  2)
  B  TORCH_LIBRARY                562 ( 15)     1089 ( 23)
  C  STABLE_TORCH_LIBRARY         670 ( 50)     1274 ( 50)

  pybind11 -> TORCH_LIBRARY           +468          +995   mandatory for the stable ABI
  TORCH_LIBRARY -> STABLE             +108          +184

IN CONTEXT (the toll is fixed; what it is a fraction of is not)
  device      numel   op host ns   toll as % of the call
  cpu          1024         2336                  16.68%
  cpu       1048576       291727                   0.16%
  cuda         1024         5999                   7.23%
  cuda      1048576         5855                   7.40%
  cuda     16777216        47708                   0.97%

  On CUDA the host cost is launch-bound and flat, so the toll does not decay
  with tensor size -- it stays ~8% until the kernel backpressures the queue.

WHAT IT BUYS
  arm                         torch.compile  profiler
  A  pybind11                   Unsupported      none
  B  TORCH_LIBRARY             fullgraph ok       yes
  C  STABLE_TORCH_LIBRARY      fullgraph ok       yes

Run this a few times: between-run drift exceeded the within-run IQR on our box.
```

## License

MIT — see [LICENSE](LICENSE).
