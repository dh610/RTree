import sys
from simulate import simulator

if __name__ == "__main__":
    
    if len(sys.argv) > 2:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        simulator(input_file_path, output_file_path)  
    elif len(sys.argv) > 1:
        input_file_path = sys.argv[1]
        simulator(input_file_path)
    else :
        simulator()
    
