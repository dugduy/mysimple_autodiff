from .CGs import *
from .numeric import *

_gradient_registry={}
class RegGrad:
    def __init__(self,op_type) -> None:
        self._optype=eval(op_type)
    def __call__(self,f):
        def wrapper(node,grad):
            raw_grad=f(node,grad)
            final_grad=[]
            for input_node,crgrad in zip(node.ops.input_nodes,raw_grad):
                if crgrad.shape!=input_node.shape: 
                        while crgrad.ndim > input_node.ndim:
                            crgrad=reduce_sum(crgrad,axis=0)
                        while crgrad.ndim < input_node.ndim:
                            crgrad=expand_dims(crgrad,axis=0)
                        for axis, size in enumerate(input_node.shape):
                            if size==1:
                                crgrad=reduce_sum(crgrad,axis=axis,keep_dims=1)
                        crgrad=tile(crgrad,reps=np.array(input_node.shape)//crgrad.shape)
                final_grad.append(crgrad)
            return final_grad
        _gradient_registry[self._optype]=wrapper
        return wrapper

@RegGrad('Neg')
def _neg_gradient(node,grad):
    return [-grad]

@RegGrad('Add')
def _add_gradient(node,grad):
    return grad,grad

@RegGrad('Sub')
def _sub_gradient(node,grad):
    return grad,-grad

@RegGrad('Mul')
def _mul_gradient(node,grad):
    a,b=node.ops.input_nodes
    return grad*b,grad*a

@RegGrad('Matmul')
def _matmul_gradient(node,grad):
    A,B=node.ops.input_nodes
    return grad@B.T,A.T@grad

@RegGrad('Div')
def _div_gradient(node,grad):
    a,b=node.ops.input_nodes
    return grad/b,-grad*a/b**2

@RegGrad('Pow')
def _pow_gradient(node,grad):
    a,b=node.ops.input_nodes
    return grad*b*a**(b-1),grad*log(a)*node

@RegGrad('Log')
def _log_gradient(node,grad):
    return [grad/node.ops.input_nodes[0]]

@RegGrad('Reduce_sum')
def _reduce_sum_gradient(node,grad):
    A=node.ops.input_nodes[0]
    output_shape=np.array(A.shape)
    if type(node.ops.axis)==tuple:
        output_shape[list(node.ops.axis)]=1
    else:
        output_shape[node.ops.axis]=1
    grad=reshape(grad,newshape=output_shape)
    return [grad]

@RegGrad('Reduce_Mean')
def _reduce_sum_gradient(node,grad):
    A=node.ops.input_nodes[0]
    output_shape=np.array(A.shape)
    if type(node.ops.axis)==tuple:
        output_shape[list(node.ops.axis)]=1
        devide_for=np.prod(np.array(A.shape)[list(node.ops.axis)])
    elif node.ops.axis is None:
        output_shape[node.ops.axis]=1
        devide_for=A.size
    else:
        output_shape[node.ops.axis]=1
        devide_for=A.shape[node.ops.axis]
    grad=reshape(grad,newshape=output_shape)
    return [grad/devide_for]

@RegGrad('Expandim')
def _expandim_gradient(node,grad):
    A=node.ops.input_nodes[0]
    return [reshape(grad,newshape=A.shape)]

@RegGrad('Tile')
def _tile_gradient(node,grad):
    A=node.ops.input_nodes[0]
    return [reshape(grad,newshape=tuple(node.ops.reps)+tuple(A.shape))]
@RegGrad('Transpose')
def _transpose_gradient(node,grad):
    reT=np.zeros(len(node.ops.new_dim_index),'int')
    for i, dim in enumerate(node.ops.new_dim_index):
        reT[dim]=i
    return [transpose(grad,new_dim_index=reT)]
@RegGrad('Maximum')
def _maximum_gradient(node,grad):
    A,B=node.ops.input_nodes
    where_a_gt_b=A>B
    return grad*where_a_gt_b.value,grad*~where_a_gt_b.value
@RegGrad('Minimum')
def _minimum_gradient(node,grad):
    A,B=node.ops.input_nodes
    where_a_lt_b=A<B
    return grad*where_a_lt_b.value,grad*~where_a_lt_b.value
@RegGrad('Reshape')
def _reshape_gradient(node,grad):
    return [reshape(grad,newshape=node.ops.input_nodes[0].shape)]
@RegGrad('Zeros_padding')
def _zeros_pad_gradient(node,grad):
    return [grad.__getitem__(node.ops.container)]
@RegGrad('GetItem')
def _getitem_gradient(node,grad):
    return [zeros_pad(grad,zeros_shape=node.ops.input_nodes[0].shape,container=node.ops.items)]
@RegGrad('AdjustNeg')
def _adneg_gradient(node,grad):
    return [adjustneg(grad,items=node.ops.items)]
@RegGrad('Abs')
def _abs_gradient(node,grad):
    where_lt_0=node.ops.input_nodes[0]<0
    return [adjustneg(grad,items=where_lt_0.value)]
@RegGrad('Sin')
def _sin_gradient(node,grad):
    return [cos(node.ops.input_nodes[0])*grad]
@RegGrad('Cos')
def _cos_gradient(node,grad):
    return [-sin(node.ops.input_nodes[0])*grad]
@RegGrad('Reduce_Max')
def _max_gradient(node,grad):
    A=node.ops.input_nodes[0]
    where_eq_max=(A==reduce_max(A,axis=node.ops.axis,keep_dims=True))
    output_shape=np.array(A.shape)
    if type(node.ops.axis)==tuple:
        output_shape[list(node.ops.axis)]=1
    else:
        output_shape[node.ops.axis]=1
    return [reshape(grad,newshape=output_shape)*where_eq_max/reduce_sum(where_eq_max,axis=node.ops.axis,keep_dims=True)]
@RegGrad('Unfold')
def _unfold_gradient(node,grad):
    A=node.ops.input_nodes[0]
    transposing=np.arange(grad.ndim)
    transposing[0],transposing[node.ops.axis]=transposing[node.ops.axis],transposing[0]
    final=transpose(zeros_like(A),new_dim_index=transposing[:-1])
    grad=transpose(grad,new_dim_index=transposing)
    transpose_x=(-1,)+np.arange(A.ndim)
    for i in range(grad.shape[0]):
        final+=zeros_pad(transpose(grad[i],new_dim_index=transpose_x),zeros_shape=final.shape,container=slice(i*node.ops.step,i*node.ops.step+node.ops.kernel_size))
    return [transpose(final,new_dim_index=transposing[:-1])]
@RegGrad('Prod')
def _pod_gradient(node:Variable,grad:Variable):
    A=node.ops.input_nodes[0]
    if not node.ops.keepdims and node.ops.axis is not None:
        node=expand_dims(node,axis=node.ops.axis)
        grad=expand_dims(grad,axis=node.ops.axis)
    return [grad*node/A]

def gradients(target_var):
    grad_dict={target_var:ones_like(target_var)}
    steps=traverse_postorder(target_var)
    seen=set()
    steps=[i for i in steps if hasattr(i.ops,'input_nodes') and not (i in seen or seen.add(i))]
    steps.reverse()
    for node in steps:
        grad_fn=_gradient_registry[node.ops.__class__]
        grads=grad_fn(node,grad_dict[node])
        for input_node,grad in zip(node.ops.input_nodes,grads):
            grad_dict[input_node]=grad_dict.get(input_node,0)+grad
    return grad_dict