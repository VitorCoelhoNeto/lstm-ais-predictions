import folium
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras

# Saves
MAP_SAVE_PATH   = 'ignoreFolder/predictionOutput/'
VESSEL_POS_FILE = 'ignoreFolder/decodedCSVsList/'

# Load paths
METRICS_PATH = "ignoreFolder/MetricsOutput/"
TEST_DATA    = "ignoreFolder/"

def createNumberMarker(color, number):
    """ 
    Create a 'numbered' icon to plot on the map
    """
    icon = folium.DivIcon(
            icon_size=(150,36),
            icon_anchor=(14,40),
            html="""<span class="fa-stack " style="font-size: 12pt" >>
                    <!-- The icon that will wrap the number -->
                    <span class="fa fa-circle-o fa-stack-2x" style="color : {:s}"></span>
                    <!-- a strong element with the custom content, in this case a number -->
                    <strong class="fa-stack-1x">
                         {:02d}  
                    </strong>
                </span>""".format(color,number)
        )
    return icon


def create_map(resultsList, currentMMSI):
    """
    Plots the map with the historic positions and the current positions, and the predicted positions
    resultsList = [mypred, true, distance]
    """

    # Correct coordinates
    coordinates = pd.read_csv(VESSEL_POS_FILE + str(currentMMSI) + '.csv', engine='python').values.astype('float32')
    coordinates = pd.DataFrame(coordinates)
    latitudesList =  coordinates[0].astype(float).tolist()
    longitudesLits = coordinates[1].astype(float).tolist()

    # The points from the correct coordinates
    points = []
    for i in range(len(latitudesList)):
        points.append([latitudesList[i],longitudesLits[i]])
    
    # The points from the correct coordinates
    points = []
    for i in range(len(latitudesList)):
        points.append([latitudesList[i],longitudesLits[i]])

    # Define map
    map = folium.Map(location=[latitudesList[0],longitudesLits[0]], zoom_start=13)

    # Create the markers with the correct points and add them to the map
    for index, lat in enumerate(latitudesList):
        folium.Marker(
                        [lat,longitudesLits[index]],
                        popup=('Position{} \n'.format(str(index+1))),
                        icon= createNumberMarker(color='blue',number=index+1)
                        ).add_to(map)
    
    # Add the lines to the correct points
    folium.PolyLine(points, color='blue', dash_array='5', opacity = '.85', tooltip='Position'
                    ).add_to(map)
    
    # Inference coordinates
    latitudesInference = []
    longitudesInference = []
    for inferenceResult in resultsList:
        latitudesInference.append(inferenceResult[0][0])
        longitudesInference.append(inferenceResult[0][1])
    
    # The points from the inference coordinates
    pointsInference = []
    for j in range(len(latitudesInference)):
        pointsInference.append([latitudesInference[j],longitudesInference[j]])
    
    # Create the markers with the correct points and add them to the map
    for indexInf, latInf in enumerate(latitudesInference):
        folium.Marker(
                        [latInf,longitudesInference[indexInf]],
                        popup=('Position{} \n'.format(str(indexInf+103))),
                        icon= createNumberMarker(color='red',number=indexInf+103)
                        ).add_to(map)
    
    # Add the lines to the inference points
    folium.PolyLine(pointsInference, color='red', dash_array='5', opacity = '.85', tooltip='Teste1'
                    ).add_to(map)
    
    # Save the map as html
    map.save(MAP_SAVE_PATH + str(currentMMSI) + '.html')

    # Save the obtained results
    with open(MAP_SAVE_PATH + str(currentMMSI) + 'Predictions.csv', 'w') as resultsFile:
        resultsFile.write("TRUELAT,TRUELON,PREDLAT,PREDLON,DISTANCE\n")
        for result in resultsList:
            resultsFile.write(  str(result[1][0])+ ',' 
                              + str(result[1][1])+ ','
                              + str(result[0][0])+ ','
                              + str(result[0][1])+ ','
                              + str(result[2]) + '\n')

    # Calculate MAE now for the predictions in Lat/Lon separately
    mae = keras.losses.MeanAbsoluteError()
    print("Latitude MAE: ", mae(latitudesList[102:], latitudesInference).numpy())
    mae = keras.losses.MeanAbsoluteError()
    print("Longitude MAE: ", mae(longitudesLits[102:], longitudesInference).numpy())

    # Calculate MAPE now for the predictions in Lat/Lon separately
    mape = keras.losses.MeanAbsolutePercentageError()
    print("Latitude MAPE: ", mape(latitudesList[102:], latitudesInference).numpy())
    mape = keras.losses.MeanAbsolutePercentageError()
    print("Longitude MAPE: ", mape(longitudesLits[102:], longitudesInference).numpy())


def read_saved_history(currentMMSI):
    """
    Reads and plots the graphs with the saved history from model training
    """

    path = METRICS_PATH + str(currentMMSI) + '.save'
    with open(path, "rb") as file_pi:
        history = pickle.load(file_pi)

    # Plot multiple graphs
    figure, axis = plt.subplots(2, 3)
    axis[0, 0].plot(history['loss'], marker='o', color='g')
    axis[0, 0].set_title("Loss")
    axis[0, 0].set_ylabel("Loss")
    axis[0, 0].set_xlabel("Epoch")

    axis[0, 1].plot(history['mean_absolute_error'], marker='o', color='r')
    axis[0, 1].set_title("MAE")
    axis[0, 1].set_ylabel("MAE")
    axis[0, 1].set_xlabel("Epoch")

    axis[0, 2].plot(history['mean_absolute_percentage_error'],marker='o', color='y')
    axis[0, 2].set_title("MAPE")
    axis[0, 2].set_ylabel("MAPE (in %)")
    axis[0, 2].set_xlabel("Epoch")

    axis[1, 0].plot(history['val_loss'], marker='o')
    axis[1, 0].set_title("Validation Loss")
    axis[1, 0].set_ylabel("Loss")
    axis[1, 0].set_xlabel("Epoch")

    axis[1, 1].plot(history['val_mean_absolute_error'], marker='o', color='k')
    axis[1, 1].set_title("Validation MAE")
    axis[1, 1].set_ylabel("MAE")
    axis[1, 1].set_xlabel("Epoch")

    axis[1, 2].plot(history['val_mean_absolute_percentage_error'], marker = 'o',color='m')
    axis[1, 2].set_title("Validation MAPE")
    axis[1, 2].set_ylabel("MAPE (in %)")
    axis[1, 2].set_xlabel("Epoch")
    plt.show()


def plot_best_score_graph(currentMMSI):
    """
    Plots the genetic algorithm best score evolution
    """

    pathToFile = METRICS_PATH + str(currentMMSI) + '.txt'
    with open(pathToFile, 'r') as file:
        bestScoreString = file.readlines()[0]
    
    # Transform string to list and convert from km to m
    bestScoreList = bestScoreString.strip('][').split(', ')
    for i, score in enumerate(bestScoreList):
        bestScoreList[i] = float(score) * 1000
    
    # Plotting the graph
    plt.plot(bestScoreList)
    plt.title('Best Score Evolution')
    plt.ylabel('Best Score')
    plt.xlabel('Generation')
    plt.legend(['score'], loc='upper left')
    plt.show()


def plot_best_score_and_history(currentMMSI):
    """
    Combines the functionalities of plot_best_score_graph and read_saved_history
    """
    
    # Genetic algorithm evolution graph
    pathToFile = METRICS_PATH + str(currentMMSI) + '.txt'
    with open(pathToFile, 'r') as file:
        bestScoreString = file.readlines()[0]
    
    # Transform string to list and convert from km to m
    bestScoreList = bestScoreString.strip('][').split(', ')
    for i, score in enumerate(bestScoreList):
        bestScoreList[i] = float(score) * 1000
    
    # Plotting the graph
    plt.plot(bestScoreList)
    plt.title('Best Score Evolution')
    plt.ylabel('Best Score')
    plt.xlabel('Generation')
    plt.legend(['score'], loc='upper left')


    # Network training graph
    path = METRICS_PATH + str(currentMMSI) + '.save'
    with open(path, "rb") as file_pi:
        history = pickle.load(file_pi)

    # Plot multiple graphs
    figure, axis = plt.subplots(2, 3)
    axis[0, 0].plot(history['loss'], marker='o', color='g')
    axis[0, 0].set_title("Loss")
    axis[0, 0].set_ylabel("Loss")
    axis[0, 0].set_xlabel("Epoch")

    axis[0, 1].plot(history['mean_absolute_error'], marker='o', color='r')
    axis[0, 1].set_title("MAE")
    axis[0, 1].set_ylabel("MAE")
    axis[0, 1].set_xlabel("Epoch")

    axis[0, 2].plot(history['mean_absolute_percentage_error'],marker='o', color='y')
    axis[0, 2].set_title("MAPE")
    axis[0, 2].set_ylabel("MAPE (in %)")
    axis[0, 2].set_xlabel("Epoch")

    axis[1, 0].plot(history['val_loss'], marker='o')
    axis[1, 0].set_title("Validation Loss")
    axis[1, 0].set_ylabel("Loss")
    axis[1, 0].set_xlabel("Epoch")

    axis[1, 1].plot(history['val_mean_absolute_error'], marker='o', color='k')
    axis[1, 1].set_title("Validation MAE")
    axis[1, 1].set_ylabel("MAE")
    axis[1, 1].set_xlabel("Epoch")

    axis[1, 2].plot(history['val_mean_absolute_percentage_error'], marker = 'o',color='m')
    axis[1, 2].set_title("Validation MAPE")
    axis[1, 2].set_ylabel("MAPE (in %)")
    axis[1, 2].set_xlabel("Epoch")

    plt.show()