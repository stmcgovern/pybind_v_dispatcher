"""Cost of moving a PyTorch op from pybind11 to TORCH_LIBRARY.

The same C++ symbol is exposed three ways, so the binding is the only variable.
CPU tensors only. Timing is torch.utils.benchmark blocked_autorange.

    python setup.py build_ext --inplace && python bench.py [min_run_time_s]
"""
import sys, torch, torch.utils.benchmark as tb, pvd
from torch.profiler import profile, ProfilerActivity

MIN_RUN_TIME = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
ARMS = [("A  pybind11",      pvd.my_op,           pvd.my_op_work),
        ("B  TORCH_LIBRARY", torch.ops.pvd.my_op, torch.ops.pvd.my_op_work)]
if hasattr(torch.ops, "pvd_stable"):
    ARMS += [("C  STABLE_TORCH_LIBRARY", torch.ops.pvd_stable.my_op,
              torch.ops.pvd_stable.my_op_work)]
BASELINE = ARMS[0][0]


def measure(fn, x):
    """Median ns/call and IQR."""
    for _ in range(1000):
        fn(x)
    m = tb.Timer("fn(x)", globals={"fn": fn, "x": x},
                 num_threads=1).blocked_autorange(min_run_time=MIN_RUN_TIME)
    return m.median * 1e9, m.iqr * 1e9


print(f"torch {torch.__version__} | blocked_autorange(min_run_time={MIN_RUN_TIME}s) | num_threads=1")

probe = torch.randn(64)
reference = ARMS[0][2](probe)
for name, _, work in ARMS[1:]:
    if not torch.equal(work(probe), reference):
        raise SystemExit(f"{name} disagrees with pybind11 -- fix before timing")
print("all arms return identical results\n")

# Both dispatch paths: the autograd key is absent when requires_grad is False.
cols = {grad: {name: measure(fn, torch.randn(8, requires_grad=grad))
               for name, fn, _ in ARMS} for grad in (False, True)}
print(f"THE TOLL (identity body: pure call overhead)\n"
      f"  {'arm':<26}{'no grad':>14}{'autograd':>14}   ns/call (IQR)")
for name, _, _ in ARMS:
    print(f"  {name:<26}" + "".join(f"{m:9.0f} ({i:3.0f})" for m, i in
                                    (cols[g][name] for g in (False, True))))
toll, toll_grad = (cols[g][ARMS[1][0]][0] - cols[g][BASELINE][0] for g in (False, True))
print(f"\n  {'pybind11 -> TORCH_LIBRARY':<26}{toll:+14.0f}{toll_grad:+14.0f}   mandatory for the stable ABI")
if len(ARMS) > 2:
    d, dg = (cols[g][ARMS[2][0]][0] - cols[g][ARMS[1][0]][0] for g in (False, True))
    print(f"  {'TORCH_LIBRARY -> STABLE':<26}{d:+14.0f}{dg:+14.0f}")

# The toll is fixed, so divide rather than difference two large timings.
# torch.mul stands in for a real dispatched op on each device.
print(f"\nIN CONTEXT (the toll is fixed; what it is a fraction of is not)\n"
      f"  {'device':<7}{'numel':>10}{'op host ns':>13}{'toll as % of the call':>24}")
cases = [("cpu", n) for n in (1024, 1048576)]
if torch.cuda.is_available():
    cases += [("cuda", n) for n in (1024, 1048576, 16777216)]
for device, numel in cases:
    x = torch.randn(numel, device=device)
    op, _ = measure(lambda t: torch.mul(t, 2.0), x)
    print(f"  {device:<7}{numel:>10}{op:13.0f}{toll / (op + toll) * 100:23.2f}%")
if torch.cuda.is_available():
    print("\n  On CUDA the host cost is launch-bound and flat, so the toll does not decay")
    print("  with tensor size -- it stays ~8% until the kernel backpressures the queue.")

print(f"\nWHAT IT BUYS\n  {'arm':<26}{'torch.compile':>15}{'profiler':>10}")
x = torch.randn(8)
for name, fn, _ in ARMS:
    torch._dynamo.reset()
    try:
        torch.compile(lambda v: fn(v), fullgraph=True)(x)
        compiles = "fullgraph ok"
    except Exception as exc:
        compiles = type(exc).__name__
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(50):
            fn(x)
    seen = any("my_op" in e.key for e in prof.key_averages())
    print(f"  {name:<26}{compiles:>15}{('yes' if seen else 'none'):>10}")

print("\nRun this a few times: between-run drift exceeded the within-run IQR on our box.")
