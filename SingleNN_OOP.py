import sys

class NN(object):

    def __init__(self):
        self.w0 = 0.7
        self.w1 = 0.3
        self.b = 0.1
        self.s = 0
        self.a = 0
        self.dw0 = 0
        self.dw1 = 0
        self.db = 0


    def forward(self, x0, x1):
        self.s = self.w0 * x0 + self.w1 * x1 + self.b
        self.a = self.s
        return self.a
    
    def compute_gradients(self, x0, x1, y):
        self.dw0 = -(y - self.a) * x0
        self.dw1 = -(y - self.a) * x1
        self.db  = -(y - self.a)

    def update_weights_biases(self, lr):
        self.w0 = self.w0 - lr * self.dw0
        self.w1 = self.w1 - lr * self.dw1
        self.b  = self.b  - lr * self.db

    def train(self, x0, x1, y, epochs=10000, lr=0.001):
        for i in range(0, epochs):
            loss = 0
            for j in range(0, len(y)):
                a = self.forward(x0[j], x1[j])
                loss += 0.5 * (y[j] - a) ** 2
                self.compute_gradients(x0[j], x1[j], y[j])
                self.update_weights_biases(lr)
            print('loss =', loss)

def main():
    x0 = [1,2,3,4,5,6,7]
    x1 = [2.2,4.5,5.6,8.6,10.15,12.44,14.23]
    y = [0,1,0,1,0,1,0]
    nn = NN()
    nn.train(x0,x1,y)

    # test
    x = 2.7
    y = 6.0
    z = nn.forward(x,y)
    print('z =', z)

if __name__ == "__main__":
    sys.exit(int(main() or 0))