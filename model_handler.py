import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tqdm import tqdm
import random
from network import Network
from genetic_algorithm import evolve
from datetime import datetime


def start_model_creation(currentMMSI):
    """
    Creates an LSTM model based on the best fitness from a genetic algorithm
    """
    print("Model creation for vessel ", currentMMSI, " initialized")
    startTime = datetime.now()
    startTime = startTime.strftime("%H:%M:%S")

    # Global variables
    dataset = 'ignoreFolder/decodedCSVsList/' + str(currentMMSI) + '.csv'
    bestScore = 100000.0
    # Number of times to evole the population
    generations = 50
    # Number of networks in each generation
    population = 15

    bestScoreEvolution = []

    # Parameters which will be used to create networks with random values on each key
    nn_param_choices = {
        'lstms':[1,2],
        'implementation1':[1,2],
        'units1':[2,8,16,32,64,128],
        'lstm_activation1':['tanh', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
        'recurrent_activation1':['hard_sigmoid', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
        'implementation2':[1,2],
        'units2':[2,8,16,32,64,128],
        'lstm_activation2':['tanh', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
        'recurrent_activation2':['hard_sigmoid', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
        'nb_layers': [1, 2, 3, 4, 5],
            'nb_neurons1': [2,8,16,32,64,128],    
            'activation1': ['tanh', 'sigmoid', 'linear', 'relu'],
            'nb_neurons2': [2,8,16,32,64,128],    
            'activation2': ['tanh', 'sigmoid', 'linear', 'relu'],
            'nb_neurons3': [2,8,16,32,64,128],    
            'activation3': ['tanh', 'sigmoid', 'linear', 'relu'],
            'nb_neurons4': [2,8,16,32,64,128],    
            'activation4': ['tanh', 'sigmoid', 'linear', 'relu'],
            'nb_neurons5': [2,8,16,32,64,128],
            'activation5': ['tanh', 'sigmoid', 'linear', 'relu'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
    }
    print(f"Evolving {generations} generations with population {population}")

    ### Generate population ###

    # Initialize the genetic algorithm and the networks that will be used
    networks = []
    for _ in range(population):
        # Create a random network.
        network = Network(nn_param_choices)
        for key in network.nn_param_choices:
            network.network[key] = random.choice(network.nn_param_choices[key])
        # Add the network to our population.
        networks.append(network)

    # Evolve the generation.
    for i in range(generations):
        print('\n\n','-'*80)
        print(f"Doing generation {i + 1} of {generations}")

        # Train and get accuracy for networks.
        progressBar = tqdm(total=len(networks))
        for j, network in enumerate(networks):
            print(f"\nDoing network {j+1} of population {population}")
            bestScore = network.train(dataset, bestScore, currentMMSI)
            progressBar.update(1)
        progressBar.close()

        # Sort the networks
        networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)

        # Check best score
        print('Best score up until now:', int(1000*bestScore))
        print('-'*80)
        bestScoreEvolution.append(bestScore)

        ## Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)
    networks[:1][0].print_network()
    
    # Metadata
    endTime = datetime.now()
    endTime = endTime.strftime("%H:%M:%S")
    print("Time started: ", str(startTime), "\nTime finished: ", str(endTime))
    
    with open('ignoreFolder/MetricsOutput/' + str(currentMMSI) + '.txt', 'w') as geneticEvolutionFile:
        geneticEvolutionFile.write(str(bestScoreEvolution))