from operations import Value
import random

class Neuron:
    def __init__(self, nin):
        self.weights =  [Value(random.uniform(1,-1)) for _ in range(nin)]
        self.bias = Value(random.uniform(1,-1))

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        output = act.tanh()
        return output
        pass

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
        pass

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin]+nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == '__main__':
    xs = [
        [2.0, 3.0, -1.0], 
        [3.0, -1.0, 0.5], 
        [0.5, 1.0, 1.0], 
        [1.0, 1.0, -1.0]
    ]
    
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(3, [4, 4, 1])
    EPOCHS = 20
    y_pred1 = [n(x) for x in xs]
    print(y_pred1)

    for k in range(EPOCHS):

        y_pred = [n(x) for x in xs]
        loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, y_pred))

        for p in n.parameters():
            p.gradient = 0.0

        loss.backward()

        for p in n.parameters():
            p.value += -0.1 * p.gradient

        print(k, loss.value)

print(y_pred)
print(n([1.9, 2.9, -0.9]))
