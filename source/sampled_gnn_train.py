# Backward-compatible entrypoint.
from train import parse_args, train

if __name__ == "__main__":
    train(parse_args())
