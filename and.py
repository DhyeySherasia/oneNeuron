from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model
import pandas as pd
import logging


logging_str = "[ %(asctime)s - %(levelname)s - %(module)s - %(message)s] %"
logging.basicConfig(level=logging.INFO, format=logging_str)


def main(data, eta, epochs, filename):

    df = pd.DataFrame(data)
    df

    x, y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)  # Will initialize the random weights
    model.fit(x, y)

    save_model(model, filename)




if __name__ == '__main__':  # --> Entry point for this file

    LEARNING_RATE = 0.3
    EPOCHS = 10

    AND = {
            "x1": [0, 0, 1, 1],
            "x2": [0, 1, 0, 1],
            "y": [0, 0, 0, 1]
        }

    main(data=AND, eta=LEARNING_RATE, epochs=EPOCHS, filename="and.model")
