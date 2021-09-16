from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd

AND = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 0, 0, 1]
}

df = pd.DataFrame(AND)
df

x, y = prepare_data(df)

LEARNING_RATE = 0.3
EPOCHS = 10

model = Perceptron(eta=LEARNING_RATE, epochs=EPOCHS)  # Will initialize the random weights
model.fit(x, y)



