import os
from kafka import KafkaConsumer
from pyais import FileReaderStream
from decoder import decode_single_message
from model_handler import start_model_creation
from predictions import predict_position
from plotting import create_map

KAFKA_SERVER      = 'localhost:9092'
KAFKA_TOPIC       = 'test'
KAFKA_API_VERSION = (0,10)
CSV_FILE_PATH     = 'outputs/decodedCSVsList/'

def start_kafka():
    """
    Start the Kafka consumer which will receive the AIS messages
    """
    
    # Create Kafka consumer and listen to server
    consumer = KafkaConsumer(KAFKA_TOPIC,bootstrap_servers=[KAFKA_SERVER], api_version=KAFKA_API_VERSION) #auto_offset_reset='earliest' to get messages since beggining of topic
    for message in consumer:

        # Get the message in AIS readable form
        with open("temp.log", 'w') as tempFile:
            tempFile.write(str(message.value.decode()))
        byteMessage = ""
        for msg in FileReaderStream("temp.log"):
            byteMessage = msg

        # Decode the message and save it to a CSV
        currentMMSI = decode_single_message(byteMessage)

        # Check if message was saved
        if not os.path.exists(CSV_FILE_PATH + str(currentMMSI)+".csv"):
            print("Unsupported message type")
        else:
            # We only want the model to be created every 100 lines, or else, predict the following position. 
            # Clean the file every 100 positions after the first 100 by deleting the first 100 positions
            # Get the number of lines and its contents
            with open(CSV_FILE_PATH+ str(currentMMSI)+".csv", 'r') as decodedFile:
                numLines = len(decodedFile.readlines())
            with open(CSV_FILE_PATH+ str(currentMMSI)+".csv", 'r') as decodedFile2:
                fileLines = decodedFile2.readlines()

            if numLines%100 == 1 and numLines > 102:
                with open(CSV_FILE_PATH + str(currentMMSI) + '.csv', 'w') as fileToWrite:
                    fileToWrite.write("LATITUDE,LONGITUDE,SPEED,HEADING\n")
                with open(CSV_FILE_PATH + str(currentMMSI) + '.csv', 'a') as fileToWrite2:
                    for line in fileLines[102:]:
                        fileToWrite2.write(line)

            # Create a model every 100 lines
            if numLines%100 == 1:
                start_model_creation(currentMMSI)
            elif numLines > 103:
                resultsList = predict_position(currentMMSI)
                create_map(resultsList, currentMMSI)

    consumer.close()
