import sys
from kafka_handler import start_kafka
from predictions import predict_position
from plotting import plot_best_score_and_history

if __name__ == "__main__":
    """
    Main function
    """

    if len(sys.argv) == 1 or len(sys.argv) >= 4:
        print("\nPlease provide one of the following commands:")
        print("\t -> \"python main.py -r\" to run the app continuously")
        print("\t -> \"python main.py -p <vesselID>\" to get next position predictions")
        print("\t -> \"python main.py -v <vesselID>\" to view result graphs")
        exit()

    if sys.argv[1] == '-r' and len(sys.argv) == 2:
        start_kafka()
        exit()

    if sys.argv[1] == '-p' and len(sys.argv) == 3:
        print("Run predictions for given vessel ID")
        predict_position(sys.argv[2])
        exit()

    if sys.argv[1] == '-v' and len(sys.argv) == 3:
        print("Run graph plotting for given vessel ID")
        plot_best_score_and_history(sys.argv[2])
        exit()

    print("\nPlease provide one of the following commands:")
    print("\t -> \"python main.py -r\" to run the app continuously")
    print("\t -> \"python main.py -p <vesselID>\" to get next position predictions")
    print("\t -> \"python main.py -v <vesselID>\" to view result graphs")
    exit()