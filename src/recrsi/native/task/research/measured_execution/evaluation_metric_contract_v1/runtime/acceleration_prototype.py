"""Isolated prototype. Never caches embeddings or gradients across optimizer steps."""
import torch
ORIGINAL_MM=torch.sparse.mm
class ConstantSparseMM(torch.autograd.Function):
 @staticmethod
 def forward(ctx,a,at,x):
  if a.requires_grad or at.requires_grad:raise ValueError('Learnable edge weights unsupported')
  ctx.save_for_backward(at)
  return ORIGINAL_MM(a,x)
 @staticmethod
 def backward(ctx,grad):
  at,=ctx.saved_tensors
  return None,None,ORIGINAL_MM(at,grad)
class TransposeCache:
 def __init__(self,csr=False):self.a=None;self.at=None;self.version=None;self.csr=csr;self.forward_a=None
 def mm(self,a,x,*args,**kwargs):
  if args or kwargs or a.layout!=torch.sparse_coo or a.requires_grad:return ORIGINAL_MM(a,x,*args,**kwargs)
  if not torch.is_grad_enabled() or not x.requires_grad:return ORIGINAL_MM(a,x)
  if self.a is not a or self.version!=a._version:
   self.a=a;self.version=a._version;self.at=a.transpose(0,1).coalesce();self.forward_a=a
   if self.csr:self.forward_a=a.to_sparse_csr();self.at=self.at.to_sparse_csr()
  return ConstantSparseMM.apply(self.forward_a,self.at,x)
 def __enter__(self):torch.sparse.mm=self.mm;return self
 def __exit__(self,*args):torch.sparse.mm=ORIGINAL_MM;self.a=self.at=None
