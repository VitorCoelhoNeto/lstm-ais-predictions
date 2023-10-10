from kafka import KafkaConsumer
from pyais import FileReaderStream
from decoder import decoded_single_message

def start_kafka():
    """
    Start the Kafka consumer which will receive the AIS messages
    """
    
    consumer = KafkaConsumer('test',bootstrap_servers=['localhost:9092'], api_version=(0,10)) #auto_offset_reset='earliest' to get messages since beggining of topic
    for message in consumer:
        with open("temp.log", 'w') as tempFile:
            tempFile.write(str(message.value.decode()))
        byteMessage = ""
        for msg in FileReaderStream("temp.log"):
            byteMessage = msg
        currentMMSI = decoded_single_message(byteMessage)
        

    consumer.close()
