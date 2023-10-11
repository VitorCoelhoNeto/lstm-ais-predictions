import os
from collections import defaultdict
from pyais import decode
from pyais.stream import FileReaderStream
from tqdm import tqdm
import regex as re

csvSavePath = 'ignoreFolder/decodedCSVsList/'

def export_messages_with_multi_part(outputPath, logFilesList):
    """
        This function takes an input path, which is the log file with the encoded AIS messages, 
        and an output path, where the decoded messages will be written as a JSON file.
        It takes the messages from the said log file, and decodes them, which will then export them to a JSON file.

        outputPath: str: JSON Export (Will be converted to JSON format if not provided in that format)
    """

    # Variables initialization
    num_messages = 0
    total_dictionary = defaultdict(dict)
    fileCount = len(logFilesList)

    # For loop to go through file and decode each message. Adjust total accordingly
    for file in tqdm(logFilesList, total = fileCount):
        num_messages = decode_given_file(file, total_dictionary, num_messages)

    # Debug
    print("Total number of messages: ", num_messages)
    
    # Save the messages to CSV
    save_dictionary_to_csv(total_dictionary, outputPath)


def decode_given_file(file, total_dictionary, num_messages):
    """
    Decodes each message in the given file, where each line represents a message, or message part
    """
    
    # 450000 as it is impossible to know beforehand how many messages are there in a file due to the multi part messages, so 450000 is an average
    for msg in tqdm(FileReaderStream(file), total=450000):
        try:
            # Decode the message
            decoded_message = msg.decode().asdict()

            # Get the timestamp (year, month, day and hour) from the file's name
            fileName = re.search(r'([^\\]+$)', file)

            date = re.search(r'(?<=\_)([0-9]{2,}-[0-9]{1,}-[0-9]{1,})(?=\_)', str(fileName))
            year = date[0][0:4]
            month = date[0][5:7]
            day = date[0][8:10]
            hour =  file[-7:-5]
                        
            # Write messages to dictionary for later JSON export (only mssg type 1)
            if decoded_message["msg_type"] == 1:
                if decoded_message["mmsi"] in total_dictionary.keys():
                    total_dictionary[decoded_message["mmsi"]].update({ list(total_dictionary[decoded_message["mmsi"]].keys())[-1] + 1 : { "Repeat" : decoded_message["repeat"], "Status": decoded_message["status"], "Turn": decoded_message["turn"], "Speed": decoded_message["speed"], "Accuracy": decoded_message["accuracy"], "Latitude": decoded_message["lat"], "Longitude": decoded_message["lon"], "Course": decoded_message["course"], "Heading": decoded_message["heading"], "Year" : year, "Month" : month, "Day" : day, "Hour" : hour, "Timestamp": decoded_message["second"], "Maneuver": decoded_message["maneuver"], "Raim": decoded_message["raim"], "Radio": decoded_message["radio"] } } )
                else:
                    total_dictionary[decoded_message["mmsi"]].update({ 0 : { "Repeat" : decoded_message["repeat"], "Status": decoded_message["status"], "Turn": decoded_message["turn"], "Speed": decoded_message["speed"], "Accuracy": decoded_message["accuracy"], "Latitude": decoded_message["lat"], "Longitude": decoded_message["lon"], "Course": decoded_message["course"], "Heading": decoded_message["heading"], "Year" : year, "Month" : month, "Day" : day, "Hour" : hour, "Timestamp": decoded_message["second"], "Maneuver": decoded_message["maneuver"], "Raim": decoded_message["raim"], "Radio": decoded_message["radio"] } } )
        
        except:
            print("Error reading message from file: ", str(file))

        # Count total number of messages
        num_messages += 1
    return num_messages


def save_dictionary_to_csv(total_dictionary, outputPath):
    """
    Saves the decoded messages dictionary to a CSV file for each vessel
    """
    for mmsi, messageList in total_dictionary.items():
        needsHeader = False
        if not os.path.exists(outputPath + str(mmsi)+".csv"):
            needsHeader = True
        with open(outputPath + str(mmsi)+".csv", 'a') as fp:
            if needsHeader:
                # Header
                fp.write("LATITUDE"
                         + "," + "LONGITUDE"
                         + "," + "SPEED"
                         + "," + "HEADING"
                         + "\n")
            # Message information
            for messageCount, message in messageList.items():
                fp.write(str(message["Latitude"]    )
                         + "," + str(message["Longitude"])
                         + "," + str(message["Speed"] )
                         + "," + str(message["Heading"]  )
                         + "\n")


def get_log_files_list(path):
    """
        Gets the list of files with the .log extension under a given directory
    """
    logFilesList = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".log"):
                logFilesList.append(os.path.join(root, file))
    return logFilesList


def decode_single_message(msg):
    """
    Decode incoming Kafka message with AIS encoded string and save it to a CSV
    """
    total_dictionary = defaultdict(dict)
    try:
        # Decode the message
        decoded_message = msg.decode().asdict()
                        
        # Write messages to dictionary for later JSON export (only mssg type 1)
        if decoded_message["msg_type"] == 1:
            if decoded_message["mmsi"] in total_dictionary.keys():
                total_dictionary[decoded_message["mmsi"]].update({ list(total_dictionary[decoded_message["mmsi"]].keys())[-1] + 1 : { "Speed": decoded_message["speed"], "Latitude": decoded_message["lat"], "Longitude": decoded_message["lon"],"Heading": decoded_message["heading"] } } )
            else:
                total_dictionary[decoded_message["mmsi"]].update({ 0 : {"Speed": decoded_message["speed"], "Latitude": decoded_message["lat"], "Longitude": decoded_message["lon"],"Heading": decoded_message["heading"] } } )
        
    except:
        print("Error reading message")
    
    save_dictionary_to_csv(total_dictionary, csvSavePath)
    
    return decoded_message["mmsi"]