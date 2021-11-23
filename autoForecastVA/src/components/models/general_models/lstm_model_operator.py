from autoForecastVA.src.components.models.general_models.lstm_model import LSTM
import numpy as np
import torch
from torch.utils.data import DataLoader

class LSTMModelOperator():
    '''Given enough information, this class can train an LSTM model and make predictions.
    It uses Adam as optimizer. The batch size can only be one.


    During training, the LSTM model receives an input of (1, seq_length, num_vars), the model sends an output of (1, seq_length, 1).
    When many_to_many=True, outputs of all time steps (1, seq_length, 1) are used to calculate loss and for back propagation,
    When many_to_many=False, output of the last time step (1, -1, 1) is used to calculate loss and for back propgation.

    During predicting, the LSTM model receives an input of (1, seq_length, num_vars), the model sends an output of (1, seq_length, 1).
    Regardless of the value of many_to_many, the prediction will be output of the last time step (1, -1, num_vars).

    Note that many_to_many=True, intuitively, trains a better performing model, but takes longer time.

    This batches data to speed up modeling training and inference
    inputs:
        hyper_parameter_value_combination: contains number_of_training_epochs, learning_rate, number_of_hidden_dimensions, dropout_rate

    '''
    def __init__(self, hyper_parameter_value_combination=None, train_input_output_tensor_list=None, test_input_output_tensor_list=None, many_to_many=False, loss_function=None, batch_size=1, generate_train_prediction=False, generate_test_prediction=True, speed_up_inference=False, early_stopping=False, early_stopping_patience=7, early_stopping_delta=0,  customized_model_layer_name_to_layer_dictionary=None, lazy=False):
        # inputs
        self.hyper_parameter_value_combination = hyper_parameter_value_combination
        self.train_input_output_tensor_list = train_input_output_tensor_list  # each tensor should have batch_size = 1; if you want to batch data to speed up training, do it only inside this class (using the batch_data function of this class).
        self.test_input_output_tensor_list = test_input_output_tensor_list
        self.many_to_many = many_to_many
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.generate_train_prediction = generate_train_prediction
        self.generate_test_prediction = generate_test_prediction
        self.speed_up_inference = speed_up_inference
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta  # how much a loss should be smaller than the smallest loss such that we consider the loss the new smalleset loss (an improvement)
        self.customized_model_layer_name_to_layer_dictionary = customized_model_layer_name_to_layer_dictionary
        self.lazy = lazy

        # outputs
        self.number_of_input_variables = train_input_output_tensor_list[0][0].shape[2]
        self.number_of_output_variables = train_input_output_tensor_list[0][1].shape[2]

        self.model = None
        self.optimizer = None

        self.train_input_output_tensor_list_for_model_training = None

        self.train_input_output_tensor_list_for_model_inferencing = None
        self.test_input_output_tensor_list_for_model_inferencing = None

        self.list_of_epoch_loss = []
        self.list_of_train_predictions = []  # set to [] to make it convienet for evaluator when generate_train = False
        self.list_of_test_predictions = None
        self.test_prediction_loss = None
        self.early_stop_criterion_is_met = None
        self.number_of_continuous_epochs_with_no_improvement = None

        if not lazy:
            self.model = LSTM(number_of_input_variables=self.number_of_input_variables, number_of_hidden_dimensions=self.hyper_parameter_value_combination['number_of_hidden_dimensions'], number_of_output_variables=self.number_of_output_variables, dropout_rate=self.hyper_parameter_value_combination['dropout_rate'], customized_model_layer_name_to_layer_dictionary=self.customized_model_layer_name_to_layer_dictionary)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_parameter_value_combination['learning_rate'])

            if batch_size > 1:
                self.train_input_output_tensor_list_for_model_training = self.create_batched_input_output_tensor_list(input_output_tensor_list=self.train_input_output_tensor_list, batch_size=self.batch_size)
            elif batch_size == 1:
                self.train_input_output_tensor_list_for_model_training = self.train_input_output_tensor_list
            else:
                raise NotImplementedError

            self.list_of_epoch_loss = self.train_model(self.train_input_output_tensor_list_for_model_training)

            if self.speed_up_inference:
                self.train_input_output_tensor_list_for_model_inferencing = self.create_batched_input_output_tensor_list(input_output_tensor_list=self.train_input_output_tensor_list, batch_size=len(self.train_input_output_tensor_list))
                self.test_input_output_tensor_list_for_model_inferencing = self.create_batched_input_output_tensor_list(input_output_tensor_list=self.test_input_output_tensor_list, batch_size=len(self.test_input_output_tensor_list))
            else:
                self.train_input_output_tensor_list_for_model_inferencing = train_input_output_tensor_list
                self.test_input_output_tensor_list_for_model_inferencing = test_input_output_tensor_list

            if self.generate_train_prediction:
                self.list_of_train_predictions = self.get_the_last_time_step_of_the_predicted_output(self.train_input_output_tensor_list_for_model_inferencing)
            if self.generate_test_prediction:
                self.list_of_test_predictions = self.get_the_last_time_step_of_the_predicted_output(self.test_input_output_tensor_list_for_model_inferencing)
            self.test_prediction_loss = self.calculate_prediction_loss(self.list_of_test_predictions, self.test_input_output_tensor_list)

    def train_model(self, train_input_output_tensor_list):
        self.model.train()
        list_of_epoch_loss = []  # records the accumulated loss in each epoch, refreshed after each epoch ends
        for i in range(self.hyper_parameter_value_combination['number_of_training_epochs']):
            epoch_loss = 0
            for train_input_tensor, train_output_tensor in train_input_output_tensor_list:  # train and test input tensor could be a batch of data in future implementation
                self.optimizer.zero_grad()
                h_0, c_0 = self.model.init_hidden(number_of_sequences_in_a_batch=train_input_tensor.shape[0])  # batch size
                predicted_train_output_tensor, _ = self.model(input_tensor=train_input_tensor, tuple_of_h_0_c_0=(h_0, c_0))  # expect input of shape (batch_size, seq_len, input_size)
                if self.many_to_many:
                    loss_of_a_batch = self.loss_function(predicted_train_output_tensor, train_output_tensor)
                else:
                    loss_of_a_batch = self.loss_function(predicted_train_output_tensor[:, -1:, :], train_output_tensor)
                loss_of_a_batch.backward()
                self.optimizer.step()
                epoch_loss += loss_of_a_batch.item()

            list_of_epoch_loss.append(epoch_loss)
            print(f'epoch: {i:3} loss: {epoch_loss:10.8f}')
            if self.early_stopping:
                self.early_stop_criterion_is_met, self.number_of_continuous_epochs_with_no_improvement = self.should_early_stop(list_of_epoch_loss=list_of_epoch_loss, early_stopping_patience=self.early_stopping_patience, early_stopping_delta=self.early_stopping_delta)
                if self.early_stop_criterion_is_met:
                    break

        return list_of_epoch_loss

    def get_the_last_time_step_of_the_predicted_output(self, input_output_tensor_list):
        '''Given a model and data, predict from each input_tensor's last time step by the number of time steps to predict ahead'''
        self.model.eval()
        list_of_predictions = []
        for input_tensor, output_tensor in input_output_tensor_list:
            with torch.no_grad():
                h_0, c_0 = self.model.init_hidden(number_of_sequences_in_a_batch=input_tensor.shape[0])
                predicted_output_tensor, _ = self.model(input_tensor, (h_0, c_0))  # expect input of shape (batch_size, seq_len, input_size)
                # prediction_of_the_last_time_step_and_the_first_variable = predicted_output_tensor[0, -1, 0]  # always take the first in a batch (because the batch size should only be 1), and the first variable (default to be output variable)
                prediction_of_the_last_time_step_and_the_first_variable = predicted_output_tensor[:, -1, 0]  # make it to one dimensional
                list_of_predictions.append(prediction_of_the_last_time_step_and_the_first_variable)

        # all_predictions = torch.cat(list_of_predictions, dim=0).numpy().tolist()  # each element is one dimensional, so just concate them on the only dimension (dim=0)
        # return [torch.tensor(i) for i in all_predictions]  # wrap each prediction in torch.tensor for backward compatibility
        return [tensor for tensor in torch.cat(list_of_predictions, dim=0)]

    def calculate_prediction_loss(self, list_of_predictions, input_output_tensor_list):
        '''
        input_output_tensor_list contains [(input_tensor, output_tensor)]
        input_tensor should be of shape (1, seq_length, num_var)
        output_tensor should be of shape (1, seq_length, 1)

        '''
        # test_output should only have one time step
        # test_output tensor's value for the first in a batch, last time step and first variable
        list_of_test_targets = [test_input_output[1][0, -1, 0] for test_input_output in input_output_tensor_list]
        prediction_loss = self.loss_function(torch.stack(list_of_predictions), torch.stack(list_of_test_targets))
        return prediction_loss

    def create_batched_input_output_tensor_list(self, input_output_tensor_list, batch_size):
        '''Create the batched version of input_output_tensor_list'''

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, input_output_tensor_list):
                self.input_output_tensor_list = input_output_tensor_list

            def __len__(self):
                return len(self.input_output_tensor_list)

            def __getitem__(self, index):
                input_tensor, output_tensor = input_output_tensor_list[index]
                return input_tensor[0,:,:], output_tensor[0,:,:]  # remove the batch dimension as it will be added by dataloader

        dataset = Dataset(input_output_tensor_list=input_output_tensor_list)
        # gen = torch.Generator()  # DataLoader consumes random state even when shuffle=False; we pass a blank generator to let it consume the geneerator's random state instead, so DataLoader won't mess with the random states
        # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, generator=gen)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        return(list(dataloader))

    @staticmethod
    def should_early_stop(list_of_epoch_loss, early_stopping_patience, early_stopping_delta):
        '''If for epoch=patience, the smallest loss so far has not been updated, we should early stop'''
        number_of_continuous_epochs_with_no_improvement = 0
        smallest_loss = np.Inf
        early_stop_criterion_is_met = False

        for loss in list_of_epoch_loss:
            if smallest_loss - loss > early_stopping_delta:
                smallest_loss = loss
                number_of_continuous_epochs_with_no_improvement = 0
            else:
                number_of_continuous_epochs_with_no_improvement += 1

        if number_of_continuous_epochs_with_no_improvement >= early_stopping_patience:
            early_stop_criterion_is_met = True
        return early_stop_criterion_is_met, number_of_continuous_epochs_with_no_improvement
