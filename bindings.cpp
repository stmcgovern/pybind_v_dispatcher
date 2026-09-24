// Arms A (pybind11) and B (TORCH_LIBRARY) over the same C++ symbols.
#include <torch/extension.h>
#include <torch/library.h>

// >>> your op goes here; keep both arms calling the same function <<<

// Zero-cost body: timing it gives call overhead alone.
at::Tensor my_op(const at::Tensor& x) { return x; }

// Real work, so cost scales with numel. float32 and contiguous only.
at::Tensor my_op_work(const at::Tensor& x) {
  TORCH_CHECK(x.scalar_type() == at::kFloat, "my_op_work: expected float32");
  TORCH_CHECK(x.is_contiguous(), "my_op_work: expected a contiguous tensor");
  auto out = at::empty_like(x);
  const float* in = x.const_data_ptr<float>();
  float* o = out.mutable_data_ptr<float>();
  for (int64_t i = 0; i < x.numel(); ++i) o[i] = in[i] * 2.0f + 1.0f;
  return out;
}

TORCH_LIBRARY(pvd, m) {
  m.def("my_op(Tensor x) -> Tensor");
  m.def("my_op_work(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(pvd, CompositeExplicitAutograd, m) {
  m.impl("my_op", TORCH_FN(my_op));
  m.impl("my_op_work", TORCH_FN(my_op_work));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("my_op", &my_op);
  m.def("my_op_work", &my_op_work);
}
