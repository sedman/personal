from deep_learning_chapter5_train import gradient_check
from deep_learning_chapter5_train import train

if __name__ == "__main__":
    print("Gradient check between numerical one and back-propagation")
    gradient_check()
    print("\nLet's train using back-propagation")
    train()

