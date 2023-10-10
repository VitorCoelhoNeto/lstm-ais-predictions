import sys
#from decoder import get_log_files_list, export_messages_with_multi_part, decoded_single_message
from kafka_handler import start_kafka

if __name__ == "__main__":
    """
    Main function
    """

    if len(sys.argv) == 1 or len(sys.argv) >= 4:
        print("\nPlease provide one of the following commands:")
        print("\t -> \"python main.py -r\" to run the app continuously")
        print("\t -> \"python main.py -p <vesselID>\" to get next position predictions")
        print("\t -> \"python main.py -v <vesselID>\" to view result graphs")
        print("\t -> \"python main.py -m <vesselID>\" to create map\n")
        exit()

    if sys.argv[1] == '-r' and len(sys.argv) == 2:
        start_kafka()
        exit()

    if sys.argv[1] == '-p' and len(sys.argv) == 3:
        print("Run predictions for given vessel ID")
        exit()

    if sys.argv[1] == '-v' and len(sys.argv) == 3:
        print("Run graph plotting for given vessel ID")
        exit()

    if sys.argv[1] == '-m' and len(sys.argv) == 3:
        print("Run map plotting for given vessel ID")
        exit()

    print("\nPlease provide one of the following commands:")
    print("\t -> \"python main.py -r\" to run the app continuously")
    print("\t -> \"python main.py -p <vesselID>\" to get next position predictions")
    print("\t -> \"python main.py -v <vesselID>\" to view result graphs")
    print("\t -> \"python main.py -m <vesselID>\" to create map\n")
    exit()