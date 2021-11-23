import torch
from torch import nn
class LSTM(nn.Module):
    '''We create an LSTM model consisting of LSTM cells and a linear layer. The hidden state from the last LSTM cell and time step will be feed into the linear layer; the linear layer performs linear transformation for the hidden state and outputs the prediction.

    This LSTM class has two functions: 1. create a model 2. specify how the model handle the inputs

    As of 1., the model is fixed to have three layers (lstm, dropout, linear), and this class provides a default way of creating each of the layer; however, users can customized each layer by providing a layer_name_to_layer_dictionary.

    '''

    def __init__(self, number_of_input_variables=1, number_of_hidden_dimensions=100, number_of_output_variables=1, dropout_rate=0.5, customized_model_layer_name_to_layer_dictionary=None):
        super().__init__()

        self.number_of_input_variable = number_of_input_variables
        self.number_of_hidden_dimensions = number_of_hidden_dimensions
        self.output_size = number_of_output_variables
        self.dropout_rate = dropout_rate
        self.customized_model_layer_name_to_layer_dictionary = customized_model_layer_name_to_layer_dictionary

        if self.customized_model_layer_name_to_layer_dictionary == None:
            # default layers
            self.lstm = nn.LSTM(input_size=number_of_input_variables, hidden_size=number_of_hidden_dimensions, batch_first=True)  # expect input of shape (batch, seq_len, input_size)
            self.dropout = nn.Dropout(dropout_rate)
            self.linear = nn.Linear(number_of_hidden_dimensions, number_of_output_variables)
        else:
            # customized model in the form of layers
            self.lstm = self.customized_model_layer_name_to_layer_dictionary['lstm']
            self.dropout = self.customized_model_layer_name_to_layer_dictionary['dropout']
            self.linear = self.customized_model_layer_name_to_layer_dictionary['linear']


    def init_hidden(self, number_of_sequences_in_a_batch):
        return (torch.zeros(1, number_of_sequences_in_a_batch, self.number_of_hidden_dimensions), torch.zeros(1, number_of_sequences_in_a_batch, self.number_of_hidden_dimensions))  #  num_layers, batch size (number of sequences), hidden_size

    def forward(self, input_tensor, tuple_of_h_0_c_0):
        '''
        :param input_tensor: an array where the length of each sequence and the number of sequences in a batch explained in the other arguments
        '''

        h_0, c_0 = tuple_of_h_0_c_0
        lstm_out, (h_0, c_0) = self.lstm(input_tensor, (h_0, c_0))  # expected input of shape (batch_size, seq_len, input_size)
        lstm_out_dropout = self.dropout(lstm_out)
        predictions = self.linear(lstm_out_dropout)  # expected input of shape (batch_size, any number of dimensions*, hidden_dim)
        return predictions, (h_0, c_0)