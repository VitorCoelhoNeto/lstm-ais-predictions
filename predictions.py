import joblib
import math
import numpy as np
import pandas as pd
from tensorflow import keras
from kafka import KafkaProducer
import geopy.distance
from latlongTodistbear import transform_dataset

# Testing Globals
g_testingDataPath    = 'ignoreFolder/decodedCSVsList/'
g_scalerXFilename    = 'ignoreFolder/ScalersOutput/'
g_scalerYFilename    = 'ignoreFolder/ScalersOutput/'
g_model_to_be_loaded = 'ignoreFolder/ModelsOutput/'


def predict_position(currentMMSI):
    """
    Position prediction
    """
    # Load the model
    model = keras.models.load_model(g_model_to_be_loaded + str(currentMMSI) + '.keras')
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})

    # Transform matrix to Distance Bearing and reshape it for the model's shape
    transformedPredictionDataset = transform_dataset(g_testingDataPath + str(currentMMSI) + '.csv', True)
    transformedPredictionDataset = np.delete(transformedPredictionDataset, -1, 0)
    
    # Load the saved scalers
    scaler_X = joblib.load(g_scalerXFilename + str(currentMMSI) + 'X.save')
    scaler_Y = joblib.load(g_scalerYFilename + str(currentMMSI) + 'Y.save')
    
    # Transform the predicted output
    transformedPredictionDataset = scaler_X.transform(transformedPredictionDataset)
    
    if not isinstance(model.get_layer(index=0), keras.layers.Dense):
        transformedPredictionDataset=transformedPredictionDataset.reshape(len(transformedPredictionDataset),1,4)
    
    # Inverse transform the predicted results for printing
    testPredict = scaler_Y.inverse_transform(model.predict(transformedPredictionDataset))

    # Reverse Conversion to Lat/Lon
    # Original dataset coordinates
    trueCoordinates = pd.read_csv(g_testingDataPath + str(currentMMSI) + '.csv', engine='python').values.astype('float32')
    # Pushes the first element of the array to last and every other element goes up 1 position, ex: [0,1,2,3,4] -> [1,2,3,4,0]
    trueCoordinates=np.roll(trueCoordinates, -1, axis=0)
    trueCoordinates = trueCoordinates[100:]

    # Get all prediction results values and differences
    resultsList = []

    # Get each prediction
    for i in range(len(trueCoordinates)-2):
    
        R = 6378.1 # Radius of the Earth in km

        # In the following lines, we have the predicted Distance/Bearing, and the current position
        # With the predicted Distance/Bearing, we can predict the next point in order to compare with the actual next position (trueCoordinates[i+1])
        dPred    = testPredict[i,0]               # Distance in km
        brngPred = testPredict[i,1]*(math.pi/180) # Bearing is 90 degrees converted to radians
        assert brngPred == math.radians(testPredict[i,1])

        lat1 = math.radians(trueCoordinates[i,0])  # Current lat point converted to radians
        lon1 = math.radians(trueCoordinates[i,1])  # Current long point converted to radians

        lat2 = math.asin( math.sin(lat1)*math.cos(dPred/R) + math.cos(lat1)*math.sin(dPred/R)*math.cos(brngPred))                      # Terminal Latitude
        lon2 = lon1 + math.atan2(math.sin(brngPred)*math.sin(dPred/R)*math.cos(lat1), math.cos(dPred/R)-math.sin(lat1)*math.sin(lat2)) # Terminal Longitude

        # The actual true value
        true   = (trueCoordinates[i+1,0],trueCoordinates[i+1,1])
        # The prediction
        mypred = (round(math.degrees(lat2), 7),round(math.degrees(lon2),7))
        # Distance between them
        distance = round(geopy.distance.geodesic(true, mypred).m, 2)

        #print("Correct:", true, "\tInference:", mypred, "\tDistance between: ", str(distance), "meters")
        resultsList.append([mypred, true, distance])

    # Calculate average predicted distances
    average = 0
    for prediction in resultsList:
        average += prediction[2]
    average = average/len(resultsList)

    # Report anomalous behavior if one exists
    for value in  resultsList:
        if value[2] >= (average * 2.5) and value[2] > 5550: # 3 nautical miles
            producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0,10,1))
            producer.send('anomalousReport', 'Anomalous Behavior Detected in vessel {0}. Predicted position: {1}, True position: {2}, Distance: {3}'.format(str(currentMMSI), str(value[0]), str(value[1]), str(value[2])).encode())

    return resultsList
