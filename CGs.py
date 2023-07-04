import numpy as np
class Graph:
    def __init__(self,name='myCGs') -> None:
        self.ops=[]
        self.vars=[]
        self.phs=[]
        self.name=name
    def as_default(self):
        global _default_graph
        _default_graph=self

# class Node:
#     def __init__(self,name='') -> None:
#         self.name=name
    # def __add__(self,other):
    #     if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
    #         other=Variable(other)
    #     return add(self,other)
    # def __radd__(self,other):
    #     return self+other
    # def __sub__(self,other):
    #     if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
    #         other=Variable(other)
    #     return sub(self,other)
    # def __rsub__(self,other):
    #     return -self+other
    # def __mul__(self,other):
    #     if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
    #         other=Variable(other)
    #     return mul(self,other)
    # def __rmul__(self,other):
    #     return self*other
    # def __matmul__(self,other):
    #     return matmul(self,other)
    # def __truediv__(self,other):
    #     if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
    #         other=Variable(other)
    #     return div(self,other)
    # def __rtruediv__(self,other):
    #     return div(Variable(other),self)
    # def __pow__(self,other):
    #     if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
    #         other=Variable(other)
    #     return pow(self,other)
    # def __rpow__(self,other):
    #     if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
    #         other=Variable(other)
    #     return pow(other,self)
    # def __neg__(self):
    #     return neg(self)
    
    # def __str__(self) -> str:
    #     return str(self.output)
    
class Operation:
    def __init__(self,input_nodes=[], name='') -> None:
        self.input_nodes=input_nodes
        self.name=name
        _default_graph.ops.append(self)
    def compute(self):
        pass

class Neg(Operation):
    def __init__(self, a, name='') -> None:
        super().__init__([a], name)
    def compute(self,a_val):
        return -a_val

class Add(Operation):
    def __init__(self, a,b, name='') -> None:
        super().__init__([a,b], name)
    def compute(self, x_val, y_val):
        return x_val+y_val

class Sub(Add):
    def compute(self, x_val, y_val):
        return x_val-y_val

class Mul(Add):
    def compute(self, x_val, y_val):
        return x_val*y_val
    
class Matmul(Add):
    def compute(self, x_val, y_val):
        return x_val@y_val

class Div(Add):
    def compute(self, x_val, y_val):
        return x_val/y_val

class Pow(Add):
    def compute(self, x_val, y_val):
        return x_val**y_val
    
class Reduce_sum(Operation):
    def __init__(self, A, axis=None, keep_dims=False, name='') -> None:
        super().__init__([A], name)
        self.axis=axis
        self.keep_dims=keep_dims
    def compute(self,A_val):
        return np.sum(A_val,self.axis,keepdims=self.keep_dims)
    
class Log(Neg):
    def compute(self, x_val):
        return np.log(x_val)

class Expandim(Operation):
    def __init__(self, A,axis, name='') -> None:
        super().__init__([A], name)
        self.axis=axis
    def compute(self,A_val):
        return np.expand_dims(A_val,self.axis)

class Tile(Operation):
    def __init__(self, A,reps, name='') -> None:
        super().__init__([A],name)
        self.reps=reps
    def compute(self,A_val):
        return np.tile(A_val,self.reps)

# class PlaceHolder(Node):
#     def __init__(self, name='') -> None:
#         super().__init__(name)
#         _default_graph.phs.append(self)

class Variable:
    def __init__(self,init_val, name='',ops=None) -> None:
        self.name=name
        self.value=np.array(init_val)
        self.ops=ops
        _default_graph.vars.append(self)
    def __add__(self,other):
        if not (type(other)==Variable):   
            other=Variable(other)
        return add(self,other)
    def __radd__(self,other):
        return self+other
    def __sub__(self,other):
        if not (type(other)==Variable):   
            other=Variable(other)
        return sub(self,other)
    def __rsub__(self,other):
        return -self+other
    def __mul__(self,other):
        if not (type(other)==Variable):   
            other=Variable(other)
        return mul(self,other)
    def __rmul__(self,other):
        return self*other
    def __matmul__(self,other):
        return matmul(self,other)
    def __truediv__(self,other):
        if not (type(other)==Variable):   
            other=Variable(other)
        return div(self,other)
    def __rtruediv__(self,other):
        return div(Variable(other),self)
    def __pow__(self,other):
        if not (type(other)==Variable):   
            other=Variable(other)
        return pow(self,other)
    def __rpow__(self,other):
        if not (type(other)==Variable):   
            other=Variable(other)
        return pow(other,self)
    def __neg__(self):
        return neg(self)
    # def __str__(self) -> str:
    #     return str(self.value)
    def __repr__(self) -> str:
        return f'Variable({self.value},name="{self.name}")'
    


def reduce_sum(A,axis=None,keep_dims=False,name=''):
    sum_obj=Reduce_sum(A,axis,keep_dims,name)
    return Variable(sum_obj.compute(A.value),name,sum_obj)

def neg(a,name=''):
    neg_obj=Neg(a,name+'_ops')
    return Variable(neg_obj.compute(a.value),name,neg_obj)

def log(a,name=''):
    log_obj=Log(a,name+'_ops')
    return Variable(log_obj.compute(a.value),name,log_obj)

def add(a,b,name=''):
    add_obj=Add(a,b,name+'_ops')
    return Variable(add_obj.compute(a.value,b.value),name,add_obj)

def sub(a,b,name=''):
    sub_obj=Sub(a,b,name+'_ops')
    return Variable(sub_obj.compute(a.value,b.value),name,sub_obj)

def mul(a,b,name=''):
    mul_obj=Mul(a,b,name+'_ops')
    return Variable(mul_obj.compute(a.value,b.value),name,mul_obj)

def div(a,b,name=''):
    div_obj=Div(a,b,name+'_ops')
    return Variable(div_obj.compute(a.value,b.value),name,div_obj)

def pow(a,b,name=''):
    pow_obj=Pow(a,b,name+'_ops')
    return Variable(pow_obj.compute(a.value,b.value),name,pow_obj)

def matmul(A,B,name=''):
    matmul_obj=Matmul(A,B,name+'_ops')
    return Variable(matmul_obj.compute(A.value,B.value),name,matmul_obj)

def expand_dims(A,axis=0,name=''):
    exp_dim_obj=Expandim(A,axis,name+'_ops')
    return Variable(exp_dim_obj.compute(A.value),name,exp_dim_obj)

def tile(A,reps,name=''):
    tile_obj=Tile(A,reps,name+'_ops')
    return Variable(tile_obj.compute(A.value),name,tile_obj)

def traverse_postorder(var_obj):
    nodes_postorder=[]
    def recurse(node):
        if hasattr(node.ops,'input_nodes'):
            for input_node in node.ops.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(var_obj)
    return nodes_postorder