import sys 
import os 
sys.path.append(os.path.join("..", "src"))

import data

def test_mnist():
    batch = next(iter(data.load_mnist(train=False, batch_size=2)))[0]
    return list(batch.size()) == [2, 1, 28, 28]


def test_cifar10():
    batch = next(iter(data.load_cifar10(train=False, batch_size=2)))[0]
    return list(batch.size()) == [2, 3, 32, 32]


def main():
    test_mnist()
    test_cifar10()
    print("ALL DATA TESTS PASSED!")

if __name__ == '__main__':
    main()