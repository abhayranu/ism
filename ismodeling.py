import tkinter as tk
from tkinter import ttk
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
import pygraphviz as pgv

##  Prompt User to Input Number of Features for the Interpretive Structural Model   ##

ele = int(input('Please Enter The Number of Features for Your ISM Model:  '))
col = list()
for e in range(ele):
    col.append(input(f'Please Enter the Name of Feature {e+1} :  '))

column_names = np.array(col)



##  Function for Binary Matrix Multiplication   ##

def bmm(A,B):
    size = len(A)
    C = np.zeros((size,size), dtype='int64')
    for i in range(size):
        for j in range(size):
            temp = A[i,:]@B[:,j]
            if temp >= 1:
                C[i][j] = 1
            else:
                C[i][j] = 0
    return C



##  Function for Element Wise Comparing Two Matrix   ##

def eq(A,B):
    rows = len(A)
    columns = len(A[0])
    for i in range(rows):
        for j in range(columns):
            if A[i][j] != B[i][j]:
                return False
    return True


##  Function for Finding the Intersection of two lists  ##

def intersection(A,B):
    i_list = list()
    for item in B:
        if item in A:
            i_list.append(item)
    return i_list




## Graphical User Interface for User to Input the Initial Matrix    ##

class MatrixGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Selection")
        self.matrix = [[tk.StringVar() for _ in range(ele)] for _ in range(ele)]

        self.row_names = column_names
        self.column_names = column_names

        self.create_matrix()
        self.create_button()

    def create_matrix(self):
        # Create column names
        for j, col_name in enumerate(self.column_names):
            col_label = ttk.Label(self.root, text=col_name, font=('Helvetica', 12, 'bold'))
            col_label.grid(row=0, column=j+1)

        # Create row names and matrix cells
        for i, row_name in enumerate(self.row_names):
            row_label = ttk.Label(self.root, text=row_name, font=('Helvetica', 12, 'bold'))
            row_label.grid(row=i+1, column=0)

            for j in range(ele):
                frame = ttk.Frame(self.root, borderwidth=1, relief=tk.RAISED, width=50, height=50)
                frame.grid(row=i+1, column=j+1)

                # Dropdown menu
                value = tk.StringVar()
                value.set('0')  # Default value '0'
                dropdown = ttk.OptionMenu(frame, value, '0', '0', '1', command=lambda v, r=i, c=j: self.on_dropdown_change(v, r, c))
                dropdown.pack(fill=tk.BOTH, expand=True)

                self.matrix[i][j] = value

    def create_button(self):
        btn = ttk.Button(self.root, text="Process", command=self.process_matrix_values)
        btn.grid(row=len(self.row_names) + 1, columnspan=len(self.column_names) + 1, padx=10, pady=10)

    def on_dropdown_change(self, value, row, col):
        self.matrix[row][col].set(value)


    def process_matrix_values(self):
        # Get the matrix values before closing the window
        self.matrix_values = [[int(cell.get()) for cell in row] for row in self.matrix]

        # Close the GUI window
        self.root.destroy()

def get_matrix_values():
    root = tk.Tk()
    app = MatrixGUI(root)
    root.mainloop()
    return app.matrix_values



##  The Main Loop   ##

if __name__ == "__main__":
    matrix_values = get_matrix_values()


    # Get the Matrix from The input values of the GUI #

    A = np.array(matrix_values)                     
    l = len(A)

    # Create the Adjacency Matrix B #

    B = A + np.eye(l, dtype='int64')


    # Loop to Find the Reachability Matrix T #

    condition = True
    B_old = B.copy()
    while condition:
        B_new = bmm(B,B_old)
        if eq(B_old, B_new):
            condition = False
        else:
            B_old = B_new.copy()
    T = B_old


    # FInding the Reachability Set, the Antecedent Set and the Intersection of the two Sets #

    R_set, A_set, I_set = [], [], []
    for i in range(l):
        Rtemp_set = []
        for j, jtem in enumerate(T[i,:]):
            if jtem==1:
                Rtemp_set.append(j+1)
        R_set.append(Rtemp_set)
        Atemp_set = []
        for k, ktem in enumerate(T[:,i]):
            if ktem==1:
                Atemp_set.append(k+1)
        A_set.append(Atemp_set)
        I_set.append(intersection(R_set[i],A_set[i]))


    # Loop Partitioning and Finding Hierarchical Level #

    L_set = np.zeros(l, dtype='int64')
    condition = True
    n = 0
    while condition:
        n += 1
        com = []
        for i, item in enumerate(R_set):
            if len(item)==1:
                L_set[i] = n
                com.append(item[0])
        for c in com:
            for j, _ in enumerate(R_set):
                if c in R_set[j]:
                    R_set[j].remove(c)
            for k, _ in enumerate(A_set):
                if c in A_set[k]:
                    A_set[k].remove(c)
        if np.prod(L_set)!=0:
            condition = False
    

    # Reduction into Canonical Matrix from the Reachability Matrix #

    CM = pd.DataFrame(T, columns=column_names, index=column_names)
    CM['Level'] = L_set.astype(dtype='int64')
    new_row = pd.DataFrame(L_set, index=column_names, columns=['Level'])
    CM = pd.concat([CM, new_row.T])
    CM = CM.sort_values(by=CM.columns[-1], axis=0)
    CM = CM.sort_values(by=CM.columns[-1], axis=1)
    RCM = CM.copy()
    n = len(column_names)
    RCM.columns = np.arange(1,n+2)
    RCM.index = np.arange(1,n+2)

    for i in range(1,n+1):
        for j in range(i,n+1):
            if (RCM.at[j+1,i]==1) and (RCM.at[j+1,n+1]>RCM.at[i,n+1]):
                L = RCM.at[j+1,n+1]+1
                break
        if L not in L_set:
            continue
        for j in range(i,n+1):
            if RCM.at[j,n+1]==L:
                break 
        for j in range(j,n+1):
            RCM.at[j,i] = 0


    RCM_new = RCM.copy()
    RCM_new.columns = CM.columns
    RCM_new.index = CM.index

    lev = np.sort(L_set)
    lev_n = []
    for i in range(1,np.max(lev)+1):
        ac = np.argwhere(lev==i).flatten()
        # print(ac)
        lev_n.append([RCM_new.columns[item] for item in ac])
    
    lev_nn = {i: lev_n[i] for i in range(len(lev_n))}


    # Initializing the Hierarchical Directed Graph Structure #

    G = nx.DiGraph()
    for i, item in enumerate(RCM_new.columns[:-1]):
        G.add_node(item, level=lev[i])


    # Finding the Reduced Canonical Matrix #

    RCM_new.drop(index='Level', inplace=True)
    RCM_new.drop(columns='Level', inplace=True)

    F = RCM_new.to_numpy(dtype='int64')

    for j in range(l-1,-1,-1):
        for k, ktem in enumerate(F[j,:j]):
            if ktem==1:
                G.add_edge(RCM_new.columns[j], RCM_new.columns[k])
    

    # Determining Position and Plotting the Hierarchical Directed Graph #
    
    pos = nx.multipartite_layout(G, subset_key='level', align='horizontal')

    plt.figure(figsize=(15,15))
    levels = lev_nn
    for level, nodes in levels.items():
        color = plt.cm.tab20(level / np.max(lev))
        nx.draw_networkx_nodes(G, pos, nodelist=nodes)
        nx.draw_networkx_labels(G, pos, labels={node: node for node in nodes})

    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
    plt.legend()
    plt.axis('off')
    plt.show()
