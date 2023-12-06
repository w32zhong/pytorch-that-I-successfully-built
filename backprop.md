## Autograd

### AutogradMeta
In `tensor_ctor` function, the `requires_grad` option is handled by:
```c++
// ./torch/csrc/utils/tensor_new.cpp
new_tensor.set_requires_grad(args_requires_grad);

class TORCH_API TensorBase {
  const TensorBase& set_requires_grad(bool requires_grad) const {
    impl_->set_requires_grad(requires_grad);
    return *this;
  }
};

// c10/core/TensorImpl.cpp
void TensorImpl::set_requires_grad(bool requires_grad) {
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();

  autograd_meta_->set_requires_grad(requires_grad, this);
}

// c10/core/TensorImpl.h
struct C10_API TensorImpl : public c10::intrusive_ptr_target {
  std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;
};

struct C10_API AutogradMetaInterface {
  virtual void set_requires_grad(
      bool requires_grad,
      at::TensorImpl* self_impl) = 0;
};
```

Note that the `AutogradMetaInterface` is a virtual interface, it will be materialized on demand.
For example, when calling the `detach_()` function, it got the chance to call `materialize_autograd_meta`:
```c++
// torch/csrc/autograd/VariableTypeManual.cpp
Tensor& detach_(c10::DispatchKeySet ks, Tensor& self) {
  auto autograd_meta = impl::materialize_autograd_meta(self);
  autograd_meta->set_requires_grad(false, self.unsafeGetTensorImpl());
  autograd_meta->grad_fn_.reset();
  autograd_meta->output_nr_ = 0;
  autograd_meta->fw_grad_.reset();
  return self;
}

// ./torch/csrc/autograd/variable.cpp
AutogradMeta* materialize_autograd_meta(const at::TensorBase& self) {
  auto p = self.unsafeGetTensorImpl();
  if (!p->autograd_meta()) {
    p->set_autograd_meta(std::make_unique<AutogradMeta>()); // create the AutogradMeta
  }
  return get_autograd_meta(self);
}

// torch/csrc/autograd/variable.h
struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  bool requires_grad_{false};

  void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl) final {
    requires_grad_ = requires_grad;
  }
};
```

### Variable
This tensor interface is actually called `Variable` for compatibility reasons:
```c++
// torch/csrc/autograd/variable.h

// `Variable` is exactly the same as `Tensor` (for backward compatibility)
using Variable = at::Tensor;
```

### Example forward -- matrix multiplication
In [hello-world.py](./hello-world.py), there is a bare-minimal forward code example:
```py
import torch
device = 'cpu'
#device = 'cuda:0'

print('1' * 100)
A = torch.tensor([[2., 3.], [1., 4.]], requires_grad=True, device=device)

print('2' * 100)
x = torch.tensor([[6.], [-5.]], requires_grad=True, device=device)

print('3' * 100)
y = A @ x

print('4' * 100)
z = y.sum()
```
the output is:
```
$ python hello-world.py
...
3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
[call] op=[aten::matmul], key=[AutogradCPU]
callUnboxedKernelFunction with unboxed
[call] op=[aten::mm], key=[AutogradCPU]
callUnboxedKernelFunction with unboxed
callUnboxedKernelFunction with unboxed
[call] op=[aten::resolve_conj], key=[CPU]
callUnboxedKernelFunction with unboxed
4444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
[call] op=[aten::sum], key=[AutogradCPU]
callUnboxedKernelFunction with unboxed
callUnboxedKernelFunction with unboxed
[call] op=[aten::sum.dim_IntList], key=[CPU]
callUnboxedKernelFunction with unboxed
[call] op=[aten::as_strided], key=[CPU]
callUnboxedKernelFunction with sym
[call] op=[aten::fill_.Scalar], key=[CPU]
callUnboxedKernelFunction with unboxed
5555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555
...
```
Apparently, the matrix multiplication `@` or `__matmul__` has invoked the operator `aten::matmul`. But how?
It must be the class method of Tensor.

Some of the implementations are inherited from `torch._C.TensorBase` and are done in python-world, but there is no `__matmul__`: 
```py
# torch/__init__.py
from ._tensor import Tensor

# torch/_tensor.py 
class Tensor(torch._C.TensorBase):
    ...
```

However, note in the TensorBase pyi stub file, the comment indicates the implementation is in `torch/csrc/autograd/python_variable.cpp`, again, it is in C++ space!
```py
# torch/_C/__init__.pyi

# Defined in torch/csrc/autograd/python_variable.cpp
class TensorBase(metaclass=_TensorMeta):
    requires_grad: _bool
    retains_grad: _bool
    def __matmul__(self, other: Any) -> Tensor: ...
    ...
```

Let's verify! In `THPVariable_initModule`, it attaches class methods from `torch::autograd::variable_methods`,
in addition, it defines `torch._C.Tensor` or `torch._C._Tensor` where the Tensor is inherited from:
```c++
// ./torch/csrc/autograd/python_variable.cpp
PyTypeObject THPVariableType = {
    PyVarObject_HEAD_INIT(
        &THPVariableMetaType,
        0) "torch._C.TensorBase", /* tp_name */
    sizeof(THPVariable), /* tp_basicsize */
    ...
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    THPVariable_properties, /* tp_getset */
    ...
};

bool THPVariable_initModule(PyObject* module) {
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPVariableType.tp_methods = methods.data();
  /* here tp_methods adds <tensor>.<method> such as tensor_foo.__matmul__ */

  PyModule_AddObject(module, "TensorBase", (PyObject*)&THPVariableType);
  PyModule_AddObject(module, "_TensorBase", (PyObject*)&THPVariableType);
  /* here THPVariable_properties adds <tensor>.<properties> such as x.grad_fn, x.grad, and x.T */
}
```

In `torch::autograd::variable_methods`, we see `__matmul__` is actually implemented by `THPVariable_matmul`.
```c++
// ./tools/autograd/templates/python_variable_methods.cpp
PyMethodDef variable_methods[] = {
  // These magic methods are all implemented on python object to wrap NotImplementedError
  {"__add__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
  ...
  {"__matmul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_matmul>), METH_VARARGS | METH_KEYWORDS, NULL},
  ...
};

// ./torch/csrc/autograd/generated/python_variable_methods.cpp
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "matmul(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  torch::PythonArgs _r = parser.parse(self_, args, kwargs, parsed_args);
  
  auto dispatch_matmul = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    return self.matmul(other);
  };
  return dispatch_matmul(self, _r.tensor(0)); /* _r.tensor(0) extract the first arg as tensor */
}

// ./build/aten/src/ATen/core/TensorBody.h
inline at::Tensor Tensor::matmul(const at::Tensor & other) const {
    return at::_ops::matmul::call(const_cast<Tensor&>(*this), other);
}

// build/aten/src/ATen/Operators_4.cpp
static C10_NOINLINE c10::TypedOperatorHandle<matmul::schema> create_matmul_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matmul::name, matmul::overload_name)
      .typed<matmul::schema>();
}

at::Tensor matmul::call(const at::Tensor & self, const at::Tensor & other) {
    // op: c10::TypedOperatorHandle<at::Tensor(const at::Tensor&, const at::Tensor&)>
    static auto op = create_matmul_typed_handle();
    return op.call(self, other);
}
```

The `matmul::schema` is defined in
```c++
// build/aten/src/ATen/ops/matmul_ops.h
struct TORCH_API matmul {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &);
  using ptr_schema = schema*;
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::matmul")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "matmul(Tensor self, Tensor other) -> Tensor")
  static at::Tensor call(const at::Tensor & self, const at::Tensor & other);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other);
};
```
so it calls `aten::matmul` operator, which is registered in (found by using gdb breakpoint at `Dispatcher::call`):
```c++
// build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
TORCH_LIBRARY_IMPL(aten, CompositeImplicitAutograd, m) {
    m.impl("matmul", TORCH_FN(wrapper_CompositeImplicitAutograd__matmul));
}

at::Tensor wrapper_CompositeImplicitAutograd__matmul(const at::Tensor & self, const at::Tensor & other) {
  return at::native::matmul(self, other);
}

// aten/src/ATen/native/LinearAlgebra.cpp
Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  at::Tensor result, unused;
  result = at::native::_matmul_impl(unused, tensor1, tensor2);
  return result;
}

// aten/src/ATen/native/LinearAlgebra.cpp
static Tensor _matmul_impl(
    Tensor& out,
    const Tensor& tensor1,
    const Tensor& tensor2) {
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();

  TORCH_CHECK(dim_tensor1 != 0 && dim_tensor2 != 0,
              "both arguments to matmul need to be at least 1D, but they are ",
              dim_tensor1, "D and ", dim_tensor2, "D");

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return tensor1.mm(tensor2); /* our example goes here! */
  }
  ...
}

// ./build/aten/src/ATen/core/TensorBody.h
inline at::Tensor Tensor::mm(const at::Tensor & mat2) const {
    return at::_ops::mm::call(const_cast<Tensor&>(*this), mat2);
}

// build/aten/src/ATen/Operators_3.cpp
static C10_NOINLINE c10::TypedOperatorHandle<mm::schema> create_mm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mm::name, mm::overload_name)
      .typed<mm::schema>();
}

at::Tensor mm::call(const at::Tensor & self, const at::Tensor & mat2) {
    static auto op = create_mm_typed_handle();
    return op.call(self, mat2);
}

// build/aten/src/ATen/ops/mm_ops.h 
struct TORCH_API mm {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &);
  using ptr_schema = schema*;
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::mm")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "mm(Tensor self, Tensor mat2) -> Tensor")
  static at::Tensor call(const at::Tensor & self, const at::Tensor & mat2);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2);
};
```

So now it is dispatching the `aten::mm` operator with `AutogradNestedTensor` key:
```c++
// torch/csrc/autograd/generated/VariableType_3.cpp

// namespace VariableType {

TORCH_LIBRARY_IMPL(aten, AutogradNestedTensor, m) {
    m.impl("mm",
           TORCH_FN(VariableType::mm)
    );
}

at::Tensor mm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);

  auto _any_requires_grad = compute_requires_grad( self, mat2 );

  std::shared_ptr<MmBackward0> grad_fn;
  if (_any_requires_grad) {
    // torch::autograd::generated::MmBackward0
    grad_fn = std::shared_ptr<MmBackward0>(new MmBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    if (grad_fn->should_compute_output(0)) {
      grad_fn->mat2_ = SavedVariable(mat2, false);
    }
    grad_fn->mat2_layout = mat2.layout();
    grad_fn->mat2_sym_sizes = mat2.sym_sizes().vec();
    grad_fn->mat2_sym_strides = strides_or_error(mat2, "mat2").vec();
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->self_layout = self.layout();
    grad_fn->self_sym_sizes = self.sym_sizes().vec();
    grad_fn->self_sym_strides = strides_or_error(self, "self").vec();
  }

  auto _tmp = ([&]() {
    return at::redispatch::mm(ks & c10::after_autograd_keyset, self_, mat2_);
  })();
  at::Tensor result = std::move(_tmp);

  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}

// build/aten/src/ATen/RedispatchFunctions.h
namespace redispatch {
    inline at::Tensor mm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {
        return at::_ops::mm::redispatch(dispatchKeySet, self, mat2);
    }
}

// build/aten/src/ATen/Operators_3.cpp
at::Tensor mm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {
    static auto op = create_mm_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat2);
}
```

This will "redispatch" to the `wrapper_CPU_mm` CPU backend handler:
```c++
// build/aten/src/ATen/RegisterCPU.cpp 
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("mm", TORCH_FN(wrapper_CPU_mm));
}

// ./build/aten/src/ATen/ops/mm_meta.h
struct TORCH_API structured_mm : public at::impl::MetaBase {
    void meta(const at::Tensor & self, const at::Tensor & mat2);
};

// build/aten/src/ATen/ops/mm_native.h 
struct TORCH_API structured_mm_out_cpu : public at::meta::structured_mm {
    void impl(const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & out);
};

// build/aten/src/ATen/RegisterCPU.cpp
struct structured_mm_out_cpu_functional final : public at::native::structured_mm_out_cpu {
    void set_output_raw_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
    }
    std::array<Tensor, 1> outputs_;
};

at::Tensor wrapper_CPU_mm(const at::Tensor & self, const at::Tensor & mat2) {
    structured_mm_out_cpu_functional op;
    op.meta(self, mat2);
    op.impl(self, mat2, op.outputs_[0]);
    return std::move(op.outputs_[0]);
}

// ./torch/include/ATen/TensorMeta.h
#define TORCH_META_FUNC(name) void structured_##name::meta
#define TORCH_IMPL_FUNC(name) void structured_##name::impl

// aten/src/ATen/native/LinearAlgebra.cpp
TORCH_META_FUNC(mm)(const Tensor & self, const Tensor & mat2) {
// i.e., structured_mm_out_cpu::meta

  // Named Tensors allow users to give explicit names to tensor dimensions.
  // See: https://pytorch.org/docs/stable/named_tensor.html
  auto names = at::namedinference::compute_matmul_outnames(self, mat2);

  // names: std::vector<at::Dimname, std::allocator<at::Dimname> >
  set_output_raw_strided(0, {self.sizes()[0], mat2.sizes()[1]}, {}, self.options(), names);
}

// aten/src/ATen/native/LinearAlgebra.cpp
TORCH_IMPL_FUNC(mm_out_cpu)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
// i.e., structured_mm_out_cpu::impl
    addmm_impl_cpu_(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
}

// aten/src/ATen/native/LinearAlgebra.cpp
static void addmm_impl_cpu_(
    Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha) {
  const auto self_sizes = self.sizes();
  auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

  at::native::resize_output(result, self_sizes);
  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  bool transpose_c = false;
  Tensor c;

  // resolve_conj: Returns a new tensor with materialized conjugation if
  // input’s conjugate bit is set to True, else returns input.
  // it will cost an operation op=[aten::resolve_conj].

  if (result_strides[0] == 1 &&
      (result_sizes[1] == 1 || result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
    transpose_c = false;
    c = result.resolve_conj();
  } else if (result_strides[1] == 1 &&
             (result_sizes[0] == 1 || result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
    std::swap(m1, m2);
    std::swap(m1_sizes, m2_sizes);
    std::swap(m1_strides, m2_strides);
    transpose_c = true;
    c = result.resolve_conj();
  } else {
    transpose_c = false;
    c = result.resolve_conj().transpose(0, 1).contiguous().transpose_(0, 1);
  }

  const int64_t m = result_sizes[transpose_c ? 1 : 0];
  const int64_t n = result_sizes[transpose_c ? 0 : 1];
  const int64_t k = m1_sizes[transpose_c ? 0 : 1];

  bool transpose_a = false;
  Tensor a;
  if (m1_strides[transpose_c ? 1 : 0] == 1 &&
      m1_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = m1.resolve_conj();
  } else if (m1_strides[transpose_c ? 0 : 1] == 1 &&
             m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = m1;
  } else {
    transpose_a = !transpose_c;
    a = m1.clone(at::MemoryFormat::Contiguous);
  }

  bool transpose_b = false;
  Tensor b;
  if (m2_strides[transpose_c ? 1 : 0] == 1 &&
      m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = m2.resolve_conj();
  } else if (m2_strides[transpose_c ? 0 : 1] == 1 &&
             m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = m2;
  } else {
    transpose_b = !transpose_c;
    b = m2.clone(at::MemoryFormat::Contiguous);
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
  const int64_t ldc = c.strides()[transpose_c ? 0 : 1];

  // Apply BLAS routine
  _AT_DISPATCH_ADDMM_TYPES(result.scalar_type(), "addmm_impl_cpu_", [&]{
      using opmath_t = at::opmath_type<scalar_t>;
      // eventually calling LAPACK library...
      at::native::cpublas::gemm(
          transpose_a ? a.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
          transpose_b ? b.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
          m, n, k,
          alpha.to<opmath_t>(),
          a.const_data_ptr<scalar_t>(), lda,
          b.const_data_ptr<scalar_t>(), ldb,
          beta.to<opmath_t>(),
          c.mutable_data_ptr<scalar_t>(), ldc);
  });

  result.copy_(c);
}
```
