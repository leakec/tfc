from tfc.utils import ComponentConstraintGraph

# Create component constraint dictionary
N = ['u','v','w']
E = [{'name':'c1','node0':'u','node1':'v'},
     {'name':'c1','node0':'v','node1':'w'},
     {'name':'c1','node0':'w','node1':'u'},
     {'name':'c2','node0':'u','node1':'v'},
     {'name':'c3','node0':'u','node1':'v'},
     {'name':'c4','node0':'v','node1':'w'},]

# Create the component constraint graphs
p = ComponentConstraintGraph(N,E)

# Save the result to the output folder
p.SaveTrees('output',savePDFs=True)

# Tell user where to find the results
print("Open the html file output/main.html to see all of the valid component constraint graphs.\n")
