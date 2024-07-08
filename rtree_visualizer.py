from rtree import RTree, RTreeNode, Rectangle
from typing import List, Dict
import matplotlib.pyplot as plt


def draw_rectangle(
    ax : plt.Axes,
    rectagnle : Rectangle,
    text : str = None
) :
    # rectagnle to pyplot patches
    x1, x2 ,y1, y2 = rectagnle.x1, rectagnle.x2, rectagnle.y1, rectagnle.y2
    square = plt.Rectangle((x1, y1), x2-x1, y2-y1, facecolor='r',
                     edgecolor='none', alpha=0.5)
    if text is not None :
        ## Write Text inside the center of the rectangle
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        plt.text(x, y, text)
    
    ## Draw Rectangle and text
    ax.add_patch(square)
    ax.text(x, y, text)
     
    return ax


def draw_rectangle_entities(
    ax : plt.Axes,
    rectangle_dict : Dict[int, Rectangle]
) :
    for key, rectangle in rectangle_dict.items() :
        draw_rectangle(ax, rectangle, text = str(key))
    return ax



def draw_node_mbr(
    ax : plt.Axes,
    node : RTreeNode,
    node_text : str = None
) :
    x1, x2, y1, y2 = node.mbr.x1, node.mbr.x2, node.mbr.y1, node.mbr.y2
    ## if node is leaf square is dashed lined
    if node.is_leaf :
        square = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linestyle='dashed')
    else :
        square = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False)
    ax.add_patch(square)
    
    if node_text is not None :
        ## Write text above the mbr
        x = (x1 + x2) / 2
        y = y2 + 0.2
        plt.text(x, y, node_text)
    
    return ax


def bfs_traverse_and_get_all_nodes(node : RTreeNode) -> List[RTreeNode] :
    all_nodes = []
    queue = [node]
    while len(queue) > 0 :
        current_node = queue.pop(0)
        all_nodes.append(current_node)
        queue.extend(current_node.children)
    return all_nodes


def draw_tree_mbr(
    ax : plt.Axes,
    root_node : RTreeNode
) :
    ## Traverse Every node in the Trees and Draw MBRs
    ## Add index as text above the MBR
    all_nodes = bfs_traverse_and_get_all_nodes(root_node)
    
    for node_idx, node in enumerate(all_nodes) :
        ax = draw_node_mbr(ax, node, "R"+str(node_idx))
    
    return ax
    
    
def visualize(
    rtree : RTree,
    rectangle_dict : Dict[int, Rectangle],
    file_path : str,
    min_x : int = None,
    min_y : int = None,
    max_x : int = None,
    max_y : int = None
) :
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    ## Fix size of the plot using min_x, min_y, max_x, max_y
    if min_x is not None :
        ax.set_xlim(min_x - 2, max_x + 2)
    if min_y is not None :
        ax.set_ylim(min_y - 2, max_y + 2)
    
    ax = draw_rectangle_entities(ax, rectangle_dict)
    ax = draw_tree_mbr(ax, rtree.root)
    
    ## Show the plot
    plt.show()
    
    ## Save the plot to file
    plt.savefig(file_path)
    
    ## Close the plot
    plt.close()