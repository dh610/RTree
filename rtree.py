from typing import List, Tuple
import math
from itertools import combinations


class Rectangle:
    
    def __init__(self, x1 : int, y1 : int, x2 : int = None, y2 : int = None, name : str = None):
        
        if x2 is None and y2 is None:
            x2, y2 = x1, y1
        elif x2 is None and y2 is not None:
            raise ValueError("invalid rectangle nor a point") 
        elif x2 is not None and y2 is None:
            raise ValueError("invalid rectangle nor a point")   
        
        ## Rearrange the coordinates x2 being greater than x1 and y2 being greater than y1
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.name = name
            
    
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    
    def to_str(self) -> str:
        if self.name is None:
            return f"Rectangle({self.x1}, {self.y1}, {self.x2}, {self.y2})"
        else :
            return f"Rectangle_{self.name}({self.x1}, {self.y1}, {self.x2}, {self.y2})"

    
    def overlap(self, other) -> bool:
        return not (self.x1 > other.x2 or self.x2 < other.x1 or self.y1 > other.y2 or self.y2 < other.y1)
    
    
    def exact_overlap(self, other) -> bool:
        return self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2
    
    
    def within(self, other) -> bool:
        return self.x1 >= other.x1 and self.y1 >= other.y1 and self.x2 <= other.x2 and self.y2 <= other.y2
    

class RTreeNode:
    
    def __init__(
        self,
        entries : List[Rectangle] = [],
        is_leaf : bool = True
    ):  
        self.entries = entries or []
        self.is_leaf = is_leaf
        self.children = []
        self.mbr = Rectangle(0, 0, 0, 0)
    
    
    def update_node_mbrs(self):
        # Update mbr using entries or children 
        if self.is_leaf:
            x1 = min(rect.x1 for rect in self.entries)
            y1 = min(rect.y1 for rect in self.entries)
            x2 = max(rect.x2 for rect in self.entries)
            y2 = max(rect.y2 for rect in self.entries)
        else :
            x1 = min(child.mbr.x1 for child in self.children)
            y1 = min(child.mbr.y1 for child in self.children)
            x2 = max(child.mbr.x2 for child in self.children)
            y2 = max(child.mbr.y2 for child in self.children)
        
        self.mbr = Rectangle(x1, y1, x2, y2)
    
    
    

class RTree:
    
    def __init__(self, max_entries=3, min_entries=1):
        self.root = RTreeNode()
        self.max_entries = max_entries
        self.min_entries = min_entries

    """
    START OF IMPLEMENTATION OF INSERT
    """

    def Enlargement(self, node : RTreeNode, rect : Rectangle):
        # Calculate original area to calculate difference
        origin = node.mbr.area()
        
        # To calculate inserted area
        node.entries.append(rect)

        # Change is_leaf when is not leaf node. Because of update_node_mbrs() method is different at leaf node or not
        isleaf_origin = node.is_leaf
        node.is_leaf = True
        node.update_node_mbrs()
        node.is_leaf = isleaf_origin
        new = node.mbr.area()

        # Recover node
        node.entries.pop()
        node.update_node_mbrs()

        return new - origin


    # Choose node to insert and its parent
    def ChooseLeaf(self, rect : Rectangle) -> List[RTreeNode]:

        nodeList = [self.root]

        while True:
            # Leaf check
            if nodeList[0].is_leaf is True:
                return nodeList

            # Traverse tree and insert node that invited to stack 
            F = None
            mininc = 10000000

            for child in nodeList[0].children:
                enlarge = self.Enlargement(child, rect)
                if enlarge < mininc:
                    mininc = enlarge
                    F = child

            nodeList.insert(0, F)

        return nodeList

    def PickSeeds(self, node : RTreeNode):
        idx = [0, 0]
        wastemax = 0

        for i, j in combinations(range(len(node.entries)), 2):
            # make temporary node to cacluate area with two rectangles
            tmpnode = RTreeNode()
            tmpnode.entries.append(node.entries[i])
            tmpnode.entries.append(node.entries[j])
            tmpnode.update_node_mbrs()

            # PS1. Caculate inefficiency of grouping entries together
            d = tmpnode.mbr.area() - node.entries[i].area() - node.entries[j].area()

            # PS2. Choose the most wasteful pair
            if d > wastemax:
                idx[0] = i
                idx[1] = j
                wastemax = d

        return idx

    # Pick index that insert to one of the split node.
    def PickIndex(self, nodes : List[RTreeNode], node : RTreeNode, avoididx):
        bestset = [-1, 0]        # 0: index, 1: area difference
        for i in range(len(node.entries)):
            if i in avoididx:
                continue
            diffarea = abs(self.Enlargement(nodes[0], node.entries[i]) - self.Enlargement(nodes[1], node.entries[i]))
            bestset = [i, diffarea] if diffarea > bestset[1] else bestset

        return bestset[0]

    # Pick node to insert
    def PickNext(self, nodes : List[RTreeNode], rect : Rectangle):
        origin = [0, 0]
        diff = [0, 0]

        # Caculate mbr's area when new node inserted
        for i in range(2):
            origin[i] = nodes[i].mbr.area()
            nodes[i].entries.append(rect)

            # Change is_leaf when is not leaf node. Because of update_node_mbrs() method is different at leaf node or not
            isleaf_origin = nodes[i].is_leaf
            nodes[i].is_leaf = True
            nodes[i].update_node_mbrs()
            nodes[i].is_leaf = isleaf_origin

            # Calculate difference
            diff[i] = nodes[i].mbr.area() - origin[i]

            # Recover node
            nodes[i].entries.pop()
            nodes[i].update_node_mbrs()

        # Compare areas to select the group
        minidx = diff.index(min(diff))

        # If area diff same, pick group that has smaller mbr area
        if diff[0] == diff[1]:
            minidx = origin.index(min(origin))
            # If mbr area same, pick front group
            minidx = 0 if origin[0] == origin[1] else minidx

        return minidx

    def SplitNode(self, node : RTreeNode):
        # Pick first entry for each group
        idx = self.PickSeeds(node)

        nodes = [RTreeNode(is_leaf = node.is_leaf, entries = [node.entries[idx[0]]]),\
                    RTreeNode(is_leaf = node.is_leaf, entries = [node.entries[idx[1]]])]
        for i in range(2):
            if node.is_leaf is False:
                nodes[i].children.append(node.children[idx[i]])

            nodes[i].update_node_mbrs()

        while len(idx) < len(node.entries):
            i = self.PickIndex(nodes, node, idx)
            idx.append(i)

            # Check if done
            arr = [len(nodes[0].entries), len(nodes[1].entries)]
            minidx = arr.index(min(arr))

            if self.min_entries == self.max_entries + 1 - arr[1 - minidx]:
                # If one group has few entries, insert all remaining entries
                nodes[minidx].entries.append(node.entries[i])
                nodes[minidx].update_node_mbrs()
                continue

            # Select entry to assign
            minidx = self.PickNext(nodes, node.entries[i])

            #Insert entry
            nodes[minidx].entries.append(node.entries[i])
            if node.is_leaf is False:
                nodes[minidx].children.append(node.children[i])
            nodes[minidx].update_node_mbrs()

        return nodes
        
    def insert(self, rect : Rectangle):
        """
        Given a rectangle 'rect', insert the rectangle into the index
        """
        # Find position for new record
        nodeList = self.ChooseLeaf(rect)
        
        # @nodeList is LIFO. First entry is child and next all entry its ancestor entry.

        # Add record to leaf node
        nodeList[0].entries.append(rect)

        # update all mbrs
        child = None
        for node in nodeList:
            if child is not None:
                idx = node.children.index(child)
                node.entries[idx] = child.mbr
            node.update_node_mbrs()
            child = node

        while len(nodeList[0].entries) > self.max_entries:
            if len(nodeList) == 1:
                # Make new root. New root will just one entry. But that's okay because split it soon.
                newroot = RTreeNode(is_leaf = False, entries = [nodeList[0].mbr])
                newroot.children.append(nodeList[0])
                newroot.update_node_mbrs()
                self.root = newroot
                nodeList.append(newroot)

            originNode = nodeList[0]
            del nodeList[0]

            # Split node
            L, LL = self.SplitNode(originNode)
            if originNode.is_leaf is True:
                L.is_leaf = True
                LL.is_leaf = True

            # remove original entry
            nodeList[0].entries.remove(originNode.mbr)
            nodeList[0].children.remove(originNode)

            # insert new entries
            nodeList[0].entries.append(L.mbr)
            nodeList[0].entries.append(LL.mbr)

            nodeList[0].children.append(L)
            nodeList[0].children.append(LL)

            for node in nodeList:
                node.update_node_mbrs()

    """
    END OF IMPLEMENTATION OF INSERT
    """
    
    """
    START OF IMPLEMENTATION OF DELETE
    """
    def ChooseLeaf_D(self, rect : Rectangle) -> List[RTreeNode]:

        que = [[self.root]] # @que is FIFO for BFS traverse.

        while len(que) > 0:
            nodeList = que.pop()    # @nodeList is LIFO

            if nodeList[0].is_leaf:
                for entry in nodeList[0].entries:
                    if rect.exact_overlap(entry):
                        # Find it
                        return nodeList
                continue

            for child in nodeList[0].children:
                if rect.within(child.mbr):
                    # If rect in mbr, insert in queue again
                    nodeList.insert(0, child)
                    que.append(nodeList)
        
        # If cannot find anything, return empty list
        return []

    def CondenseTree(self, nodeList : List[RTreeNode]):
        N = nodeList.pop(0)
        Q = []

        # Terminate condition to recursive function
        if len(nodeList) == 0:
            # Must be root
            return Q

        # P is parent of N
        P = nodeList[0]

        if len(N.entries) >= self.min_entries:
            return Q

        # Eliminate under-full node
        P.children.remove(N)
        P.update_node_mbrs()
        Q.append(N)
        
        # Update entries
        P.entries.pop()
        for i in range(len(P.children)):
            P.entries[i] = P.children[i].mbr

        # Recursive call
        Q += self.CondenseTree(nodeList)

        return Q

    def RecursiveInsert(self, node : RTreeNode):
        # Terminate condition to recursive function
        if node.is_leaf:
            # If node is leaf, insert every entry in tree
            for entry in node.entries:
                self.insert(entry)

        else:
            for child in node.children:
                # Recursive call
                self.RecursiveInsert(child)
            
    def delete(self, delete_rectangle : Rectangle) -> List[Rectangle]:
        """
        Given a deleting_rectangle 'delete_rectangle', delete the index record with that same rectangle
        if successful, return the deleted index record
        if not successful, return None
        """
        # Find node to delete entry
        nodeList = self.ChooseLeaf_D(delete_rectangle)
        return_list = []

        if len(nodeList) == 0:
            return None

        # Delete entry
        for rect in nodeList[0].entries:
            if delete_rectangle.exact_overlap(rect):
                nodeList[0].entries.remove(rect)
                return_list.insert(0, rect)
                break

        # Update mbr
        prev = None
        for node in nodeList:
            if prev is not None:
                idx = node.children.index(prev)
                node.entries[idx] = node.children[idx].mbr
            node.update_node_mbrs()
            prev = node

        # Invoke CondenseTree
        CondenseList = self.CondenseTree(nodeList)

        # Condense tree -> Re-insert orphaned entries
        for node in CondenseList[::-1]:
            # Use @RecursiveInsert to insert every entry in tree
            self.RecursiveInsert(node)
        
        # Shorten tree
        if self.root.is_leaf is False:
            if len(self.root.children) == 1:
                self.root = self.root.children[0]

        return return_list

    """
    END OF IMPLEMENTATION OF DELETE
    """
    
    """
    START OF IMPLEMENTATION OF SEARCH
    """
    def recursive_search(self, node : RTreeNode, search_rectangle : Rectangle):
        """
         Find @search_rectangle in subtree @node
        """
        bucket = []

        # Terminate condition to recursive function
        if node.is_leaf:
            # Add leaf rectangle to bucket
            for rect in node.entries:
                if rect.within(search_rectangle):
                    bucket.append(rect)

        else:
            for child in node.children:
                if child.mbr.overlap(search_rectangle):
                    # Recursive call
                    bucket += self.recursive_search(child, search_rectangle)

        return bucket

    def search(self, search_rectangle : Rectangle) -> List[Rectangle]:
        """
         Given a rectangle 'search_rectangle', find all index records whoose
            rectangles overlap with 'search_rectangle'
        """
        return self.recursive_search(self.root, search_rectangle)
        
    """
    END OF IMPLEMENTATION OF SEARCH
    """
    
    """
    START OF IMPLEMENTATION OF PRINT    
    """
    def print_tree(self) -> str:
        """
            print node mbrs traversing bfs
            print each level mbrs
            [Rectangle(n,n,n,n)]-[Rectagle(n,n,n,n), Rectangle(n,n,n,n)]-[Rectagle(n,n,n,n), Rectangle(n,n,n,n)]
        """
        que = [[self.root, self.root.mbr, 0]]      # List of [RTreeNode, Rectangle, int(level)] for BFS search
        ret = "["

        while len(que) > 0:
            node, rect, level = que.pop(0)
            ret += rect.to_str()

            if node is None:
                if len(que) == 0:
                    ret += "]"
                else:
                    ret += ", "
                continue

            if node.is_leaf is False:
                for child in node.children:
                    que.append([child, child.mbr, level + 1])
            else:
                for entry in node.entries:
                    que.append([None, entry, level + 1])

            if len(que) > 0 and que[0][2] > level:
                ret += "]-["

            else:
                ret += ", "

        return ret
    """
    END OF IMPLEMENTATION OF PRINT    
    """
