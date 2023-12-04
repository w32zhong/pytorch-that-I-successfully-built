## Autograd
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
but the `AutogradMetaInterface` is a virtual interface.

This interface is actually instantiated by `Variable`:
```c++
// torch/csrc/autograd/variable.h
struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  bool requires_grad_{false};

  void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl) final {
    requires_grad_ = requires_grad;
  }
};
```
