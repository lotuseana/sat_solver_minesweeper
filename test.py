import cpmpy as cp

def collect():
    print(x.value())
    print()
    """"
    for row in x:
        for element in row:
            print(element.value(), end=" ") 
        print() 
    print()
    solutions.append(x.value())
    """
    



x = cp.boolvar(shape=(10, 10), name="x")
s = cp.SolverLookup.get("pysat")

constraints = []

s.add(x[1,1]==1)
n = s.solve()
print(s.status())
print(x.value())

s.add(x[3,3]+x[2,2]==1)
m = s.solveAll()
print(s.status())
print(x.value())

"""
for i in range(3):
    s += (x[i,i]==1)
    constraints.append(x[i,i]==1)
    print(constraints)

    solutions=[]
    n = s.solveAll(display=collect)
    print("from n:")
    print(n)
    print("from solutions:")
    print(len(solutions))
"""