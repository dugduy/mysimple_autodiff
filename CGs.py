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

class Node:
    def __init__(self,name='') -> None:
        self.name=name
        self.consumers=[]
    
    def __add__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
            other=Variable(other)
        return add(self,other)
    def __radd__(self,other):
        return self+other
    def __sub__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
            other=Variable(other)
        return sub(self,other)
    def __rsub__(self,other):
        return -self+other
    def __mul__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
            other=Variable(other)
        return mul(self,other)
    def __rmul__(self,other):
        return self*other
    def __matmul__(self,other):
        return matmul(self,other)
    def __truediv__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
            other=Variable(other)
        return div(self,other)
    def __rtruediv__(self,other):
        return div(Variable(other),self)
    def __pow__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
            other=Variable(other)
        return pow(self,other)
    def __rpow__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):   
            other=Variable(other)
        return pow(other,self)
    def __neg__(self):
        return neg(self)
    
    def __str__(self) -> str:
        return str(self.output)
    
class Operation(Node):
    def __init__(self,input_nodes=[], name='') -> None:
        super().__init__(name)
        self.input_nodes=input_nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)
        _default_graph.ops.append(self)
    def compute(self):
        pass

class neg(Operation):
    def __init__(self, a, name='') -> None:
        super().__init__([a], name)
    def compute(self,a_val):
        return -a_val

class add(Operation):
    def __init__(self, a,b, name='') -> None:
        super().__init__([a,b], name)
    def compute(self, x_val, y_val):
        return x_val+y_val

class sub(add):
    def compute(self, x_val, y_val):
        return x_val-y_val

class mul(add):
    def compute(self, x_val, y_val):
        return x_val*y_val
    
class matmul(add):
    def compute(self, x_val, y_val):
        return x_val@y_val

class div(add):
    def compute(self, x_val, y_val):
        return x_val/y_val

class pow(add):
    def compute(self, x_val, y_val):
        return x_val**y_val

class PlaceHolder(Node):
    def __init__(self, name='') -> None:
        super().__init__(name)
        _default_graph.phs.append(self)

class Variable(Node):
    def __init__(self,init_val, name='') -> None:
        super().__init__(name)
        self.value=init_val
        _default_graph.vars.append(self)
    def __str__(self) -> str:
        return str(self.value)
    # def setto(self,new_value):
    #     _default_graph.vars.remove(self)
    #     return Variable(new_value)
    
def traverse_postorder(op):
    nodes_postorder=[]
    def recurse(node):
        if isinstance(node,Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(op)
    return nodes_postorder

class Session:
    def run(self,op,feed_dict={}):
        self.nodes_postorder=traverse_postorder(op)
        for node in self.nodes_postorder:
            if type(node)==PlaceHolder:
                node.output=feed_dict[node]
            elif type(node)==Variable:
                node.output=node.value
            else:
                node.inputs=[i.output for i in node.input_nodes]
                node.output=node.compute(*node.inputs)
            
            if type(node.output)==list:
                node.output=np.array(node.output)
        
        return op.output


# Graph().as_default()

# x=PlaceHolder('x')
# y=2*x-1

# sess=Session()

# print(sess.run(y,{x:19}))