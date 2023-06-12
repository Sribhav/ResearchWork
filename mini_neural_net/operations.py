"""
This file contains all the classes required to run the neural network
"""
import math

class Value:
    def __init__(self, value, _children=(), _op='', label=''):
        self.value = value
        self._prev = set(_children)
        self._op = _op
        self.gradient = 0.0
        self._backward = lambda: None
        self.label = label
    
    def __str__(self):
        return f"Value: {self.value}"
    
    __repr__ = __str__

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(
            self.value + other.value,
            _children=(self, other), 
            _op="+")
        
        def _backward():
            self.gradient += 1 * output.gradient
            other.gradient += 1 * output.gradient

        output._backward = _backward
        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(
            self.value * other.value,
            _children=(self, other),
            _op="*")
        
        def _backward():
            self.gradient += other.value * output.gradient
            other.gradient += self.value * output.gradient

        output._backward = _backward
        return output
    
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __neg__(self):
        return self*-1
    
    def __sub__(self, other):
        return self + (-other)
    
    def tanh(self):
        val = self.value
        tan = (math.exp(2*val)-1)/(math.exp(2*val)+1)
        output = Value(tan, _children=(self, ), _op='tanh')

        def _backward():
            self.gradient += (1 - tan**2) * output.gradient
        output._backward = _backward

        return output
    
    def exp(self):
        val = self.value
        output = Value(math.exp(val), _children=(self, ), _op='exp')

        def _backward():
            self.gradient += output.gradient * output.value

        return output
    
    def __pow__(self, other):
        assert isinstance(other, (int,float))
        output = Value(self.value**other, (self, ), _op=f"**{other}")

        def _backward():
            self.gradient += (other * self.value**(other-1)) *output.gradient

        output._backward = _backward
        return output
    

    def backward(self):
        topo = []
        visited = set()
        def depth_first_search(v):
            visited.add(v)
            for child in v._prev:
                depth_first_search(child)
            topo.append(v)

        depth_first_search(self)

        self.gradient = 1.0
        for node in reversed(topo):
            node._backward()


    # Helper functions
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
    
    def __rtruediv__(self, other):
        return self * (other**-1)
    
    def __rsub__(self, other):
        return self + (-other)

if __name__ == '__main__':
    a = Value(-2.0, label='a')
    b = Value(3.0, label='b')

    d = a*b; d.label = 'd'
    e = a+b; e.label = 'e'
    f = d*e; f.label = 'f'
    g = f.tanh(); g.label = 'g'

    g.backward()

    for child in g._prev:
        print(child.label, child.gradient)

    print(a.gradient, b.gradient)