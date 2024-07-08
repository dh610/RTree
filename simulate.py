import sys
from io import StringIO

from rtree import RTree, Rectangle, RTreeNode
from rtree_visualizer import visualize
from collections import defaultdict


def capture_stdout_to_string(func):
    def wrapper(*args, **kwargs):
        # Save the original stdout
        original_stdout = sys.stdout

        # Create a StringIO object to capture the output
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the function with the provided arguments and keyword arguments
        func(*args, **kwargs)

        # Restore the original stdout
        sys.stdout = original_stdout

        # Get the captured output as a string
        output_string = captured_output.getvalue()

        # Return the captured output
        return output_string

    return wrapper


def compare_min_max_xys(
    new_rectangle : Rectangle,
    min_x : int,
    min_y : int,
    max_x : int,
    max_y : int
) :
    if new_rectangle.x1 < min_x :
        min_x = new_rectangle.x1
    if new_rectangle.x2 > max_x :
        max_x = new_rectangle.x2
    if new_rectangle.y1 < min_y :
        min_y = new_rectangle.y1
    if new_rectangle.y2 > max_y :
        max_y = new_rectangle.y2
    
    return min_x, min_y, max_x, max_y


def simulator(
    input_file_path : str = None,
    output_file_path : str = None
) :
    #setIO(input_file_path, output_file_path)
    if input_file_path:
        sys.stdin = open(input_file_path, 'r')
    if output_file_path:
        sys.stdout = open(output_file_path, "w")
    myRTree = None
    
    ## For Visualization
    rectangle_key = 0
    rectangle_dict = defaultdict()
    total_min_x = 0
    total_min_y = 0
    total_max_x = 0
    total_max_y = 0
    
    while (True):
        comm = sys.stdin.readline()
        comm = comm.replace("\n", "")
        params = comm.split()
        if len(params) < 1:
            continue
        print(comm)
        
        
        if params[0] == "INIT":
            max_entries = int(params[1])
            if len(params) >= 2 :
                min_entries = int(params[2])
                myRTree = RTree(max_entries = max_entries, min_entries = min_entries)
            else :
                myRTree = RTree(max_entries = max_entries, min_entries = 1)
        
            
        elif params[0] == "EXIT":
            return
        
            
        elif params[0] == "INSERT":
            inserting_rectangle = Rectangle(
                x1 = int(params[1]),
                y1 = int(params[2]),
                x2 = int(params[3]),
                y2 = int(params[4]),
                name = rectangle_key
            )
            ## Insertion
            myRTree.insert(inserting_rectangle)
            
            ## for visualization purposes
            total_min_x, total_min_y, total_max_x, total_max_y = compare_min_max_xys(
                new_rectangle = inserting_rectangle,
                min_x = total_min_x,
                min_y = total_min_y,
                max_x = total_max_x,
                max_y = total_max_y
            )
            rectangle_dict[rectangle_key] = inserting_rectangle
            rectangle_key += 1
        
            
        elif params[0] == "DELETE":
            deleting_rectangle = Rectangle(
                x1 = int(params[1]),
                y1 = int(params[2]),
                x2 = int(params[3]),
                y2 = int(params[4]),
                name = None
            )
            
            deletion = myRTree.delete(deleting_rectangle)
            
            if len(deletion) > 0 :
                deleted_entries_str = ""
                for deleted_entry in deletion :
                    ## for visualization purposes
                    deleted_entries_str += str(deleted_entry.name) + ","
                    rectangle_dict.pop(int(deleted_entry.name))
                print("DELETED " + deleted_entries_str[:-1])
            else :
                print("NOT FOUND")  
        
                  
        elif params[0] == "SEARCH":            
            search_rectangle = Rectangle(
                x1 = int(params[1]),
                y1 = int(params[2]),
                x2 = int(params[3]),
                y2 = int(params[4]),
                name = None
            )
            search_result = myRTree.search(search_rectangle)
            search_result_str = ""
        
            for searched_entry in search_result :
                search_result_str += str(searched_entry.name) + ","
            
            print("FOUND " + search_result_str[:-1])
            
            
        elif params[0] == "VIEW" :
            visualize(rtree = myRTree,
                      rectangle_dict = rectangle_dict,
                      file_path = "output.png",
                      min_x = total_min_x,
                      min_y = total_min_y,
                      max_x = total_max_x,
                      max_y = total_max_y)
        
        
        elif params[0] == "PRINT" :
            print(myRTree.print_tree())
        
        
        elif params[0] == "SEP":
            print("-------------------------")



    
    