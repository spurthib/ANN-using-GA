import numpy as np
import pickle
import random
import matplotlib.pyplot

f = open("inputs.pkl", "rb")
#loading the inputs pickle file which has the image feature to inputs
inputs = pickle.load(f) 
f.close()

f = open("outputs.pkl", "rb")
#loading the outputs pickle file which has the class labels to outputs
outputs = pickle.load(f)
f.close()

#defining reLu activation function
def relu(input):
    result = input
    result[input < 0] = 0
    return result
#defining sigmoid activation function
def sigmoid(input):
    return 1.0/(1.0 + np.exp(-1 * input))
#defining the predicted outputs function 
def predicted_output(weights,inputs,outputs,activation="relu"):
    prediction = np.zeros(shape=inputs.shape[0])
    #index for each image in 1962 images
    for index in range(inputs.shape[0]):
        #gets the value of each input for all indices
        x = inputs[index,:]
        for w in weights:
            #calculates matrix multiplication XiWi
            x = np.matmul(x,w)
            if activation == "relu":
                x = relu(x)
            elif activation == "sigmoid":
                x = sigmoid(x)
        #checking the predicted label where x value is maximum
        predicted_label = np.where(x == np.max(x))[0][0]
        #getting the index position for the predicted output
        prediction[index] = predicted_label
    #getting the actual output for the index of the predicted label
    actual_output = np.where(prediction == outputs)[0].size
    #calculating fitness for each chromosome in the population
    accuracy = (actual_output/outputs.size)*100
    return accuracy, prediction
#fitness function is defined to calculate fitness for all chromosomes in the population
#fitness here is the ratio of correctly determined data to total number of data
def fitness(weights,inputs,outputs,activation):
    accuracy = np.zeros(shape=weights.shape[0])
    for index in range(weights.shape[0]):
        current_matrix = weights[index,:]
        accuracy[index], _ = predicted_output(current_matrix,inputs,outputs,activation=activation)
    return accuracy
#defining matrix_to_vector function to convert the weights in matrix format to vector format
#vector format is used in genetic algorithm for representing weights as chromosomes in single vector
def matrix_to_vector(matrix_weights):
    vector_weights = []
    for sol_idx in range(matrix_weights.shape[0]):
        curr_vector = []
        #we have 3 matrices(2hidden + 1output) that has to be converted to vector format
        for layer_idx in range(matrix_weights.shape[1]):
            vectorweights = np.reshape(matrix_weights[sol_idx, layer_idx], newshape=(matrix_weights[sol_idx, layer_idx].size))
            curr_vector.extend(vectorweights)
        #all the 3 vectors are appended to the vector_weights
        vector_weights.append(curr_vector)
    return np.array(vector_weights)
#defining vector_to_matrix function to convert the weights in vector format to matrix format
#this function is the reverse of the above function matrix_to_vector
#we use vector_to_matrix function to calculate matrix multiplication XiWi
def vector_to_matrix(vector_weights, matrix_weights):
    matrixweights = []
    for sol_idx in range(matrix_weights.shape[0]):
        start = 0
        end = 0
        for layer_idx in range(matrix_weights.shape[1]):
            #calculating where the vector should end for each matrix conversion
            end = end + matrix_weights[sol_idx, layer_idx].size
            #gets the vector from specified start till end
            curr_vector = vector_weights[sol_idx, start:end]
            mat_layer_weights = np.reshape(curr_vector, newshape=(matrix_weights[sol_idx, layer_idx].shape))
            matrixweights.append(mat_layer_weights)
            #the next matrix starts from the index where the first matrix ended in the vector.
            start = end
    return np.reshape(matrixweights, newshape=matrix_weights.shape)
# defining function to determine the best partner for mating based on fitness
def select_mating_partners(population, fitness, num_of_parents):
    #creating the empty array to hold parents determined based on fitness
    parents = np.empty((num_of_parents, population.shape[1]))
    for parent in range(num_of_parents):
        #getting the index value where fitness is maximum
        max_fitness_index = np.where(fitness == np.max(fitness))
        max_fitness_index = max_fitness_index[0][0]
        #adding the fitness value to parents for the above determined index where fitness is maximum
        parents[parent, :] = population[max_fitness_index, :]
        #assigning the lowest value to the above found index to avoid repeation of the same parent
        fitness[max_fitness_index] = 0
    return parents
#defining crossover function to create new offsprings
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    #defining the crossover point at the center for each parent
    crossover_point = np.uint32(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        #selecting two parents to crossover
        #defining index where the first parent should crossover
        parent1_index = k%parents.shape[0]
        #defining index where the second parent should crossover
        parent2_index = (k+1)%parents.shape[0]
        #offspring will have its first parent genes from index 0 to the crossover point which is half
        offspring[k, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
         #offspring will have its second parent genes from the crossover point till the end
        offspring[k, crossover_point:] = parents[parent2_index, crossover_point:]
    return offspring
#defining mutation function to change the gene value randomly
def mutation(offspring, mutation_percent):
    num_of_mutations = np.uint32((mutation_percent*offspring.shape[1])/100)
    #mutates single gene in each offspring randomly
    mutation_indices = np.array(random.sample(range(0, offspring.shape[1]), num_of_mutations))
    for i in range(offspring.shape[0]):
        #determining the random value on the range -1.0 to +1.0
        random_value = np.random.uniform(-1.0, 1.0, 1)
        #the random value generated above is added to the gene
        offspring[i, mutation_indices] = offspring[i, mutation_indices] + random_value
    return offspring
#defining population size
no_of_chromosome = 8
#defining number of mating partners
num_parents_mating = 4
#defining generations
generations = 100
#defining mutation percentage
mutation_percent = 10
#creating initial population
initial_population_weights = []
for i in np.arange(0, no_of_chromosome):
    HL1_neurons = 150
    #randomly generating the weights between input to hiddenlayer1 neurons
    input_HL1_weights = np.random.uniform(low=-0.1, high=0.1, size=(inputs.shape[1], HL1_neurons))
    HL2_neurons = 60
    #randomly generating the weights between hiddenlayer1 to hiddenlayer2 neurons
    HL1_HL2_weights = np.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))
    output_neurons = 4
    #randomly generating the weights between hiddenlayer2 to output neurons
    HL2_output_weights = np.random.uniform(low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))
    #appending all the wights into the initial population
    initial_population_weights.append(np.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights]))
#assigning the initial population weights to the matrix_weights in the form of an array 
matrix_weights = np.array(initial_population_weights)
#converting weights in matrix format to the vector format
vector_weights = matrix_to_vector(matrix_weights)
best_outputs = []
accuracies = np.empty(shape=(generations))
for generation in range(generations):
    print("Generation: ", generation)
    #converting weights in vector format to matrix format for each chromosome
    matrix_weights = vector_to_matrix(vector_weights, matrix_weights)
    #for each chromosome, we are calculating the fitness
    Fitness = fitness(matrix_weights, inputs, outputs, activation="sigmoid")
    accuracies[generation] = Fitness[0]
    print("Fitness: ")
    print(Fitness)
    #Based on the fitness, we are selecting the parents for mating
    parents = select_mating_partners(vector_weights, Fitness.copy(), num_parents_mating)
    print("Parents: ")
    print(parents)
    #creating an offspring by doing crossover
    offspring_crossover = crossover(parents, offspring_size=(vector_weights.shape[0]-parents.shape[0], vector_weights.shape[1]))
    print("Crossover: ")
    print(offspring_crossover)
    #Adding random values to the genes using mutation
    offspring_mutation = mutation(offspring_crossover, mutation_percent=mutation_percent)
    print("Mutation: ")
    print(offspring_mutation)
    #creating new population based on offspring and parents
    vector_weights[0:parents.shape[0], :] = parents
    vector_weights[parents.shape[0]:, :] = offspring_mutation
    matrix_weights = vector_to_matrix(vector_weights, matrix_weights)
    best_weights = matrix_weights [0, :]
    acc, predictions = predicted_output(best_weights, inputs, outputs, activation="sigmoid")
    print("Fitness after " +str(generation+1) + " generations is " +str(acc))
#generating a plot with Generations and Fitness
matplotlib.pyplot.plot(accuracies, linewidth=5, color="black")
matplotlib.pyplot.xlabel("Generations", fontsize=20)
matplotlib.pyplot.ylabel("Fitness", fontsize=20)
matplotlib.pyplot.xticks(np.arange(0, generations+1, 10), fontsize=15)
matplotlib.pyplot.yticks(np.arange(0, 101, 10), fontsize=15)
##loading the final weights into the weights pickle file
pickle.dump(matrix_weights, open("weights.pkl","wb"))
#loading the iweights pickle file
f = open("weights.pkl", "rb")
weights = pickle.load(f)
f.close()
#loading the test_inputs pickle file which has the testdata to inputs
f = open("test_inputs.pkl", "rb")
test_inputs = pickle.load(f)
f.close()
#loading the test_outputs pickle file which has the test class labels to outputs
f = open("test_outputs.pkl", "rb")
test_outputs = pickle.load(f)
f.close()
#testing the accuracy for the test data
test_weights = weights [0, :]
test_acc, test_predict = predicted_output(test_weights, test_inputs, test_outputs, activation="sigmoid")
print("Test Accuracy: ",test_acc)



















