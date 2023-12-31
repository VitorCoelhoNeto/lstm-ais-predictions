# AIS message based next vessel positions predictor using an LSTM network
This project has been developed in the context of the dissertation of the master's thesis in computer engineering for ISEP, within the context of the OVERSEE project developed at Critical Software

## Setting up Kafka
To setup Kafka, download it from its [official website](https://kafka.apache.org/downloads). Select the "Scala 2.13", under the Binary Downloads.

After downloading the file, extract the compressed file in the computer's root (e.g. C:/)

Rename the folder to "kafka"

##### For windows:
Go to the "kafka" folder, then enter "config" folder, and edit the "server" file with any txt editing tool (e.g. Notepad++)

Search (CTRL + F) for "logs" and edit the "log.dirs" variable "tmp" portion to your folder path (e.g. c:/kafka/kafka-logs)
Repeat this process for the "zookeeper" file in the same "config" folder, search for the "dataDir" variable and replace the "tmp" to the relevant path (e.g. c:/kafka/zookeeper)

The default port for the server is 9092, and the server will run on "localhost:9092"

With a command line interface open on the installed Kafka directory:

To start, first run Zookeeper:
`.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties`

Then Kafka:

`.\bin\windows\kafka-server-start.bat .\config\server.properties`

Then create the "test" topic:

`kafka-topics.bat --create --bootstrap-server localhost:9092 --topic test`

Repeat this step for the "anomalousReport" topic

To stop running the server run these commands in order:

`.\bin\windows\kafka-server-stop.bat .\config\server.properties`

`.\bin\windows\zookeeper-server-stop.bat .\config\zookeeper.properties`

##### Helpful video:
[Install Apache Kafka on Windows PC](https://www.youtube.com/watch?v=BwYFuhVhshI)

## Running the app
To run the app continuously call the main function with:

`python main.py -r`

This runs the application and makes it wait for a Kafka data record from the relevant topic ("test").

This will produce a model for each vessel every 100 positions, with the relevant outputs for each received message for each relevant vessel.
Note: For the first 100 positions, no predictions nor outputs are created due to insufficient data for that given vessel.
These outputs are stored under the folder "outputs" and include:

1. decodedCSVsList: CSV with the continuous and dynamically updated vessel position, where the file name is the vessel's MMSI;
2. MetricsOutput: contains 2 files for each vessel, named with the vessel's MMSI. One is a Pickle .save file, with the model's training history metrics (MAE, MAPE and Loss), and the other is a .txt file, with the best score evolution throughout the genetic model's generations. This score is measured in meters, where the lowest score equals the best score.
3. ModelsOutput: keras file with the model produced for the vessel, where the file name is the vessel's MMSI;
4. predictionOutput: CSV with the vessel's MMSI + "Predictions" which stores the predicted positions for that given vessel, as well as an HTML file, also named after the vessel's MMSI, with an interactive map depicting the predicted positions with a red color, as well as the true positions, depicted with a blue color.
5. ScalersOutput: Pickle .save file with the relevant X and Y scalers, which represent the scalers used during the training of the model, for the relevant vessel's model, where the file name is the vessel's MMSI.

If an anomalous position is reported, a message will be produced and a report on the Kafka topic "anomalousReport" will be issued. on the relevant defined bootstrap_server.

## Other app running options
To run the predictions for an already existing vessel:

`python main.py -p <vesselID>`

This assumes an already existing model for that given vessel, and will predict the vessel's positions, with the beforementioned outputs being created in consequence of this.

To run the visualization of a model's history and genetic model evolution:

`python main.py -p <vesselID>`

This assumes an already existing model for the given vessel and will plot the graphs for the given vessel's genetic algorithm score evolution, as well as the trained model's training history metrics. 