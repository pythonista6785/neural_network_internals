import sys
def main():
    # y = 2X + 0.3
    # x0, x1 correspnd to the x,y coordinate of a point
    # if the given output is below the y = 2x + 0.3 line, the Neural Network is 
    # to output a 0, if the point is above the line, it's output is to be 1

    # create some training data
    x0 =  [1,2,3,4,5,6,7,8,9,10]
    x1 = [2.2,4.5,5.6,8.6,10.15,12.44,14.23,16.2,18.4,20.4]
    y = [0,1,0,1,0,1,0,0,1,1]   # expected output 

    # initialize weights and biases
    w0 = 0.1
    w1 = -0.23
    b = 0.22

    # Train the Single Neuron Network 
    batch_size = 5
    for i in range(0,50000):
        loss = 0
        for j in range(0, len(y) // batch_size):
            dw0 , dw1, db = 0, 0, 0
            for k in range(0, batch_size):
                index = j * batch_size + k    # selecting first five indexes in first iteration 
                a = w0 * x0[index] + w1 * x0[index] + b   # forward pass
                loss += 0.5 * (y[index] - a) ** 2         # compute loss
                dw0 += -(y[index] - a) * x0[index]
                dw1 += -(y[index] - a) * x1[index]
                db += -(y[index] - a)
            w0 = w0 - 0.001 * dw0/batch_size
            w1 = w1 - 0.001 * dw1/batch_size        # update weights, biases after 
            b = b - 0.001 * db/batch_size           # accumalating gradients in a batch
        print("Loss =", loss)

    # test for unknown data, on the trained network 
    x0 = 2.7
    x1 = 6.0
    output = x0 * w0 + x1 * w1 + b
    print('output for(',x0,',',x1,')= ', output)

    x0 = 5.3 # x coord. of point
    x1 = 10.4 # y coord. of point
    output = x0*w0 + x1*w1 + b
    print('output for (',x0,',',x1,')= ',output)



if __name__ == "__main__":
    sys.exit(int(main() or 0))