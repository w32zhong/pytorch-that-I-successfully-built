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

print('5' * 100)
z.backward(retain_graph=True)

#print('6' * 100)
#print(A.grad)
#
#print('7' * 100)
#print(x.grad)
