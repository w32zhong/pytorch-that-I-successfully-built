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

Dispatching again ...
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
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, mat2 );
  
  [[maybe_unused]] auto _any_has_forward_grad_result = (isFwGradDefined(self) || isFwGradDefined(mat2));
  std::shared_ptr<MmBackward0> grad_fn;
  if (_any_requires_grad) {
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
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::mm(ks & c10::after_autograd_keyset, self_, mat2_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat2__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(mat2_))
    TORCH_INTERNAL_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(mat2_))
    TORCH_INTERNAL_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  if (result.has_storage() && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result)) {
    TORCH_INTERNAL_ASSERT(result.storage().use_count() == 1, "function: mm");
  }
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: mm");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  c10::optional<at::Tensor> result_new_fw_grad_opt = c10::nullopt;
  if (_any_has_forward_grad_result && (result.defined())) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_tensor = toNonOptTensor(self);
      auto self_t = (self_t_raw.defined() || !self_tensor.defined())
        ? self_t_raw : at::_efficientzerotensor(self_tensor.sizes(), self_tensor.options());
      auto self_p = toNonOptPrimal(self);
      auto mat2_t_raw = toNonOptFwGrad(mat2);
      auto mat2_tensor = toNonOptTensor(mat2);
      auto mat2_t = (mat2_t_raw.defined() || !mat2_tensor.defined())
        ? mat2_t_raw : at::_efficientzerotensor(mat2_tensor.sizes(), mat2_tensor.options());
      auto mat2_p = toNonOptPrimal(mat2);
      result_new_fw_grad_opt = at::mm(self_t, mat2_p) + at::mm(self_p, mat2_t);
  }
  if (result_new_fw_grad_opt.has_value() && result_new_fw_grad_opt.value().defined() && result.defined()) {
    // The hardcoded 0 here will need to be updated once we support multiple levels.
    result._set_fw_grad(result_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ false);
  }
  return result;
}
```
