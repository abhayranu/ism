##  Import Modules  ##

import tkinter as tk
from tkinter import ttk
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt



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


    A = np.array(matrix_values)                     # Get the Matrix from The input values of the GUI
    l = len(A)                                      
    B = A + np.eye(l, dtype='int64')                # Create the Adjacenty Matrix B


    ##  Loop to Find the Reachability Matrix T  ##

    condition = True
    B_old = B
    while condition:
        B_new = bmm(B,B_old)
        if eq(B_old, B_new):
            condition = False
        else:
            B_old = B_new
    T = B_old




    ##  Finding the Reachability Set, the Antecedent Set and the Intersection of the two sets   ##

    R_set, A_set, I_set = list(), list(), list()
    for i in range(l):
        Rtemp_set = list()
        for j, jtem in enumerate(T[i,:]):
            if jtem==1:
                Rtemp_set.append(j+1)
        R_set.append(Rtemp_set)
        Atemp_set = list()
        for k, ktem in enumerate(T[:,i]):
            if ktem==1:
                Atemp_set.append(k+1)
        A_set.append(Atemp_set)
        I_set.append(intersection(R_set[i],A_set[i]))




    ##  Finding the Hierarchical Level from the Reacheability set and the Antecedent set    ##

    L_set = np.zeros(l, dtype='int64')
    condition = True
    n = 0
    while condition:
        n += 1
        com = list()
        for i, item in enumerate(R_set):
            if len(item)==1:
                L_set[i] = n
                com.append(item[0])
        # print(com)
        for c in com:
            for j, _ in enumerate(R_set):
                if c in R_set[j]:
                    R_set[j].remove(c)
            for k, _ in enumerate(A_set):
                if c in A_set[k]:
                    A_set[k].remove(c)
        if np.prod(L_set)!=0:
            condition = False

    

    ##  Creating Hierarchical Graph Structure   ##

    G = nx.DiGraph()
    for item in column_names:
        G.add_node(item)


    ##  Connecting Nodes Based on the Reachability Matrix According to the Hierarchical Level   ##

    while np.max(L_set)!=0:
        m_h = np.argwhere(L_set==np.amax(L_set)).flatten()
        for m in m_h:
            for i, item in enumerate(B[m,:]):
                if item==1:
                    if i!=m:
                        G.add_edge(column_names[m],column_names[i])
            L_set[m] = 0



    ##  Manipulation of Graph Structure to Make it Presentable  ##

    pos = graphviz_layout(G, prog='dot')

    center_x = sum(x for x, y in pos.values()) / len(pos)
    center_y = sum(y for x, y in pos.values()) / len(pos)

    for node in pos:
        x, y = pos[node]
        dx = x - center_x
        dy = y - center_y
        pos[node] = (center_x - dx, center_y - dy)

    for node in pos:
        pos[node] = (pos[node][0] * -1, pos[node][1])



    ##  Plotting the Hierarchical Directed Graph for the ISM of a Service Robot ##

    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue',
            arrowsize=20, font_size=9, font_weight='bold', 
            connectionstyle='arc')
    plt.title('Hierarchical Directed Graph(ISM) - Service Robot', fontsize=15)
    plt.axis('off')
    plt.show()
