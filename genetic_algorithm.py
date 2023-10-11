from network import Network
import random

RETAIN_PERCENTAGE = 0.4
SELECT_PERCENTAGE = 0.1
MUTATE_PERCENTAGE = 0.1

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
    

def fitness(network):
    """
    Return network accuracy
    """
    return network.accuracy


def mutate(network):
    """
    Mutate some offspring alelles
    """
    # Choose a random key.
    mutation = random.choice(list(nn_param_choices.keys()))
    # Mutate one of the params.
    network.network[mutation] = random.choice(nn_param_choices[mutation])
    return network


def breed(mother, father):
    """
    Breed two networks to generate offspring
    """
    offspring = []
    for _ in range(2):
        child = {}
        # Loop through the parameters and pick params for the kid.
        for param in nn_param_choices:
            child[param] = random.choice([mother.network[param], father.network[param]])
        # Now create a network object.
        network = Network(nn_param_choices)
        network.create_set(child)
        # Randomly mutate some of the offspring.
        if MUTATE_PERCENTAGE > random.random(): # 20%
            network = mutate(network)
        offspring.append(network)
    return offspring


def evolve(population):
    """
    Evolve each population from a given generation
    """
    # Get scores for each network.
    graded = [(fitness(network), network) for network in population]

    # Sort scores
    graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]

    # Get the number we want to keep for the next gen.
    retain_length = int(len(graded)*RETAIN_PERCENTAGE)

    # The parents are every network we want to keep.
    parents = graded[:retain_length]

    # For those we aren't keeping, randomly keep some anyway.
    for individual in graded[retain_length:]:
        if SELECT_PERCENTAGE > random.random():
            parents.append(individual)

    # Now find out how many spots we have left to fill.
    parents_length = len(parents)
    desired_length = len(population) - parents_length
    offspring = []

    # Add offspring, which are bred from two remaining networks.
    while len(offspring) < desired_length:

        # Get random parents
        male   = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)

        # Check that they aren't the same network
        if male != female:
            male   = parents[male]
            female = parents[female]

            # Breed them.
            offspringList = breed(male, female)

            # Add the offspring one at a time.
            for child in offspringList:
                # Don't grow larger than desired length.
                if len(offspring) < desired_length:
                    offspring.append(child)

    parents.extend(offspring)
    return parents
  