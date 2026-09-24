// Arm C (STABLE_TORCH_LIBRARY): same bodies, stable-ABI types.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
using torch::stable::Tensor;

Tensor my_op_stable(Tensor x) { return x; }

// const_data_ptr() is untyped here, so the checks are the only guard.
Tensor my_op_work_stable(Tensor x) {
  STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Float,
                  "my_op_work: expected float32");
  STD_TORCH_CHECK(x.is_contiguous(), "my_op_work: expected a contiguous tensor");
  Tensor out = torch::stable::empty_like(x);
  const float* in = static_cast<const float*>(x.const_data_ptr());
  float* o = static_cast<float*>(out.mutable_data_ptr());
  for (int64_t i = 0; i < x.numel(); ++i) o[i] = in[i] * 2.0f + 1.0f;
  return out;
}

STABLE_TORCH_LIBRARY(pvd_stable, m) {
  m.def("my_op(Tensor x) -> Tensor");
  m.def("my_op_work(Tensor x) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(pvd_stable, CompositeExplicitAutograd, m) {
  m.impl("my_op", TORCH_BOX(&my_op_stable));
  m.impl("my_op_work", TORCH_BOX(&my_op_work_stable));
}
