from queue import Queue
from CGs import *
_gradient_registry={}
class RegGrad:
    def __init__(self,op_type) -> None:
        self._optype=eval(op_type)
    def __call__(self,f):
        _gradient_registry[self._optype]=f

@RegGrad('neg')
def _neg_gradient(op,grad):
    return -grad

@RegGrad('add')
def _add_gradient(op,grad):
    return grad,grad
    # a,b=op.inputs
    # grad_wrt_a=grad_wrt_b=grad
    # while np.ndim(grad_wrt_a)>len(a.shape):
    #     grad_wrt_a=np.sum(grad_wrt_a)
    # for axis,size in enumerate(a.shape):
    #     if size==1:
    #         grad_wrt_a=np.sum(grad_wrt_a,axis,keepdims=1)

    # while np.ndim(grad_wrt_b)>len(b.shape):
    #     grad_wrt_b=np.sum(grad_wrt_b)
    # for axis,size in enumerate(b.shape):
    #     if size==1:
    #         grad_wrt_b=np.sum(grad_wrt_b,axis,keepdims=1)
    
    # return grad_wrt_a,grad_wrt_b

@RegGrad('sub')
def _sub_gradient(op,grad):
    return grad,-grad

@RegGrad('mul')
def _mul_gradient(op,grad):
    a,b=op.inputs
    return grad*b,grad*a

@RegGrad('matmul')
def _matmul_gradient(op,grad):
    A,B=op.inputs
    return grad@B.T,A.T@grad

@RegGrad('div')
def _div_gradient(op,grad):
    a,b=op.inputs
    return grad/b,-grad*a/b**2

@RegGrad('pow')
def _pow_gradient(op,grad):
    a,b=op.inputs
    # return grad*b*a**(b-1),1#,grad*op.output*np.log(a)
    return grad*b*a**(b-1),grad*op.output*np.log(a)

@RegGrad('log')
def _log_gradient(op,grad):
    return grad/op.inputs[0]

@RegGrad('reduce_sum')
def _reduce_sum_gradient(op,grad):
    A=op.inputs[0]
    output_shape=np.array(A.shape)
    output_shape[op.axis]=1
    # tile_scaling=A.shape//output_shape
    grad=np.reshape(grad,output_shape)
    return grad


def compute_gradients(op,steps=[]):
    grad_table={op:np.ones_like(op.output,'float32')}
    q=Queue()
    # visited={op}
    q.put(op)
    while not q.empty():
        node = q.get()
        if node!=op:
            grad_table[node]=0.
            for consumer in node.consumers:
                if consumer in steps:
                    lossgrad_wrt_input=_gradient_registry[consumer.__class__](consumer,grad_table.get(consumer,0))
                    if len(consumer.input_nodes)>1:
                        lossgrad_wrt_input=lossgrad_wrt_input[consumer.input_nodes.index(node)]
                    # hậu kỳ part
                    if lossgrad_wrt_input.shape!=node.output.shape:
                        while lossgrad_wrt_input.ndim > node.output.ndim:
                            lossgrad_wrt_input=np.sum(lossgrad_wrt_input,0)
                        while lossgrad_wrt_input.ndim < node.output.ndim:
                            lossgrad_wrt_input=np.expand_dims(lossgrad_wrt_input,0)
                        for axis, size in enumerate(lossgrad_wrt_input.shape):
                            if size==1:
                                lossgrad_wrt_input=np.sum(lossgrad_wrt_input,axis,keepdims=1)
                        lossgrad_wrt_input=np.tile(lossgrad_wrt_input,np.array(node.output.shape)//lossgrad_wrt_input.shape)
                    grad_table[node]+=lossgrad_wrt_input.astype('float32')
                    # print(node.name,lossgrad_wrt_input)
        
        if hasattr(node,'input_nodes'):
            for i in node.input_nodes:
                # if i not in visited:
                    # visited.add(i)
                    q.put(i)
    return grad_table