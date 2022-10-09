from tfc.utils import ComponentConstraintGraph
from tfc.utils.TFCUtils import ComponentConstraintDict

# Create component constraint dictionary
N = ['u','v','w']
E = [ComponentConstraintDict(name='c1',node0='u',node1='v'),
     ComponentConstraintDict(name='c1',node0='v',node1='w'),
     ComponentConstraintDict(name='c1',node0='w',node1='u'),
     ComponentConstraintDict(name='c2',node0='u',node1='v'),
     ComponentConstraintDict(name='c3',node0='u',node1='v'),
     ComponentConstraintDict(name='c4',node0='v',node1='w'),]

# Create the component constraint graphs
p = ComponentConstraintGraph(N,E)

# Save the result to the output folder
p.SaveGraphs('output',savePDFs=True)

# Tell user where to find the results
print("Open the html file output/main.html to see all of the valid component constraint graphs.\n")
