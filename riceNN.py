import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# Basically is the program that we'll use to train the data
# If cuda and backends isn't available, the cpu is used by default
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# The neural network class
# Input Node Size: 7 Input
# Output Node Size: 1 Output (What type of rice it is)
# Hidden/Output Activation: Sigmoid 
# Input: Linear combination
class NeuralNetwork(nn.Module):
    def __init__(self, input_val, hidden_layers):
        super().__init__() # Inheriting the previous initialzied values from nn.Module
        self.input = nn.Linear(input_val, hidden_layers[0]).to(torch.float32)
        # nn.Module is basically python's list that can store modules.
        self.hidden = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]).to(torch.float32) for i in range(len(hidden_layers) - 1)])
        self.output = nn.Linear(hidden_layers[-1], 1).to(torch.float32)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.input(x))
        # Iterate through the hidden layer, where the Linear input (item) passes through the activation function Sigmoid
        for items in self.hidden:
            x = self.sigmoid(items(x))
        x = self.output(x)
        return x


df = pd.read_csv("riceclass.csv")

df_input = df.drop(["id", "Class"], axis = 1)
df_output = df["Class"]

# Apply label encoding
label_encoder = LabelEncoder()
df_output = label_encoder.fit_transform(df_output)

# Min Max Scalar
scale = MinMaxScaler(feature_range=(0, 1))
input_rescale = scale.fit_transform(df_input)
df_input = pd.DataFrame(data = input_rescale, columns = df_input.columns)


# 
inputTensor = torch.tensor(df_input.to_numpy(), dtype = torch.float32)
outputTensor = torch.tensor(df_output, dtype = torch.float32)

# Train Test Split (80% Train)
tensorDf = torch.utils.data.TensorDataset(inputTensor, outputTensor)
train_amount = int(0.8 * len(df))
test_amount = len(df) - train_amount
train_df, test_df = torch.utils.data.random_split(tensorDf, [train_amount, test_amount])

# Set up our Neural network variables
input_size = 7
hidden_layer_size = [32, 128, 32]
learning_rate = 0.1
amount_epochs = 1000

# Create the model ()
model = NeuralNetwork(input_size, hidden_layer_size)

# Set up loss functions for backward propagation and optimizeers
loss = nn.BCEWithLogitsLoss() 
optimize = optim.SGD(model.parameters(), lr = learning_rate, weight_decay=1e-5)

# Batch SGD
train_data_loader = DataLoader(train_df, batch_size = 300, shuffle = True)

for epochs in range(amount_epochs):
    for input_val, output_val in train_data_loader:
        optimize.zero_grad()
        # Use the model with the input data, where result are the outputs.
        result = model(input_val)

        # Back propagation with optimizer
        '''
        if epochs == amount_epochs - 1:
            print("ADFGASFAE")
            print(output_val.view(-1,1))
            print(result)
            print("ADFGASFAE")
        '''
        # view transposes the output_val for comparison
        lossResult = loss(result, output_val.view(-1,1))
        lossResult.backward()
        optimize.step()

    # Print progress
    if (epochs + 1) % 100 == 0:
        print(f'Epoch [{epochs+1}/{amount_epochs}], Loss: {lossResult.item():.4f}')



