from network_functions import train_network

class Network():

    def __init__(self, nn_param_choices=None):
        '''
        Initializes the Network with its default parameters:
            - accuracy: The accuracy of the network which will be used to decided whether it is fit enough or not
            - nn_param_choices: dict of random parameters to be chosen from to create the network
            - network: the network parameters
        '''
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}

    def create_set(self, network):
        self.network = network

    def train(self, dataset, bestScore, currentMMSI):
        '''
        Train the network with a dataset
        '''
        if self.accuracy == 0.:
            self.accuracy, bestScore = train_network(self.network, dataset, bestScore, currentMMSI)
        return bestScore
  
    def print_network(self):
        """
        Shows network information
        """
        print("Network error: %.2f m" % (1000*self.accuracy))
        print("%d LSTM layers" % self.network['lstms'])
        if (self.network['lstms']>0):
            print("first LSTM layer: ", self.network['units1'], self.network['lstm_activation1'], self.network['recurrent_activation1'], self.network['implementation1'])
            if (self.network['lstms']>1):	
                print("second LSTM layer: ", self.network['units2'], self.network['lstm_activation2'], self.network['recurrent_activation2'], self.network['implementation2'])	
        print("First Hidden Layer: %d neurons, with" % self.network['nb_neurons1'], self.network['activation1'], "activation")
        if (self.network['nb_layers']>1):
            print("Second Hidden Layer: %d neurons, with" % self.network['nb_neurons2'], self.network['activation2'], "activation")
        if (self.network['nb_layers']>2):
            print("Third Hidden Layer: %d neurons, with" % self.network['nb_neurons3'], self.network['activation3'], "activation")
        if (self.network['nb_layers']>3):
            print("Fourth Hidden Layer: %d neurons, with" % self.network['nb_neurons4'], self.network['activation4'], "activation")
        if (self.network['nb_layers']>4):
            print("Fifth Hidden Layer: %d neurons, with" % self.network['nb_neurons5'], self.network['activation5'], "activation")
        print('Optimizer: ', self.network['optimizer'])
        print('-'*80)
        