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

### Forward
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
```
