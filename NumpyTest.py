import sys
import numpy as np

def main():
    d1 = np.zeros((10,2))  # create a 10X2 array, initialize with zeros 
    #print(d1)

    a1 = np.zeros(4)       # create an array or vector and initialize to zeros 
    #print(a1)

    a2 = [3,4,7,9]
    a2np = np.array(a2)    # pass a2 list to a numpy array constructor 
    #print(a2np)
    sum1 = np.sum(a2np)    # add all elements of a2np array 
    #print('sum1= ', sum1)

    a3 = [2,7,9,6]
    a3np = np.array(a3)
    #print(a3)
    prodsum = np.dot(a2np,a3np) # do a dot product of two arrays 
    #print('prodsum= ',prodsum)

    elm_prod = np.multiply(a2np, a3np) #element  by element multiplication
    #print('elm_prod= ', elm_prod)

    a2v = a2np.reshape(len(a2np), 1)  # convert 1-D array to nx1 vector 
    a3v = a3np.reshape(len(a3np), 1)  # convert 1-D array to nx1 vector
    #print('a2v= ', a2v)
    #print('a3v=', a3v)

    matres = np.dot(a2v, a3v.T)   # will do nx1 and 1xn matrix multiplication 
    #print(matres)

    #matrix multiplication of 2x3 with 3x2 matrix 
    m1 = np.array([[1,4,3], [4,5,2]])
    m2 = np.array([[1,2], [2,3], [3,6]])
    m3 = np.dot(m1,m2)
    print(m3)



if __name__ == "__main__":
    sys.exit(int(main() or 0))