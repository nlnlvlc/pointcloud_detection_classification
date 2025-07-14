import numpy as np
from scipy.io import loadmat
import pandas as pd

def groundtruthConv():

    matFile = ('data/SUNRGBDtoolbox/Metadata/groundtruth.mat')
    varName = 'groundtruth'
    # Load the .mat file containing the MATLAB struct
    data = loadmat(matFile)
    key_rmv = {'__header__', '__version__', '__globals__', }
    data = {k: v for k, v in data.items() if k not in key_rmv}
    #print(f"Data Keys: {data.keys()}")

    # Access the struct (assuming 'your_struct_name' is the variable name in MATLAB)
    groundtruth = data[varName][0]
    names = groundtruth.dtype.names
    # print(f"Groundtruth: {groundtruth}")
    #print(f"Names: {names}\nLength: {len(groundtruth)}\nFirst: {groundtruth[0]}")

    # Convert the structured NumPy array to a Python dictionary
    # This often involves iterating through the fields and handling nested structures
    python_dict = {name: [] for name in groundtruth.dtype.names}

    #print(f"Python Dict: {python_dict}")

    for x in groundtruth:
        for i in range(len(names)):
            python_dict[names[i]].append(np.squeeze(x[i]))

    #df = pd.DataFrame({"Fields": python_dict.keys(), "Values": python_dict.values()})
    df = pd.DataFrame(python_dict)

    df.to_csv(f"{varName}.csv", index=False)
    df.to_excel(f"{varName}.xlsx", index=False)

    #print(df.head(10))

    #for name in names: print(f"Updated {name}: {python_dict[name][0]}")

def allsplitConv():
    file = 'data/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'
    varName = 'allsplit'
    # Load the .mat file containing the MATLAB struct
    data = loadmat(file)
    key_rmv = {'__header__', '__version__', '__globals__', }
    data = {k: v for k, v in data.items() if k not in key_rmv}
    #print(f"Data Keys: {data.keys()}")

    #Access the struct (assuming 'your_struct_name' is the variable name in MATLAB)
    names = ['alltrain', 'alltest', 'trainsplit', 'valsplit']
    alltrain = data['alltrain'][0]
    alltest = data['alltest'][0]
    trainsplit = data['trainvalsplit']['train'][0][0]
    valsplit = data['trainvalsplit']['val'][0][0]

    dataset = [alltrain, alltest, trainsplit, valsplit]

    # Convert the structured NumPy array to a Python dictionary
    # This often involves iterating through the fields and handling nested structures
    python_dict = {'alltrain': list(alltrain),
                   'alltest':list(alltest),
                   'trainsplit':list(trainsplit),
                   'valsplit':list(valsplit)
                   }

    #print(f"Keys: {type(python_dict['alltrain'])}")
    #df = pd.DataFrame({"Fields": python_dict.keys(), "Values": python_dict.values()})
    for name in names:
        df = pd.DataFrame(python_dict[name])
        df.to_csv(f"{name}.csv", index=False, header=False)
        df.to_excel(f"{name}.xlsx", index=False, header=False)


    #for name in names: print(f"Updated {name}: {python_dict[name][0]}")

def segConv(matFile, varName):
    # Load the .mat file containing the MATLAB struct
    data = loadmat(matFile)
    key_rmv = {'__header__', '__version__', '__globals__', }
    data = {k: v for k, v in data.items() if k not in key_rmv}
    #print(f"Data Keys: {data.keys()}")

    # Access the struct (assuming 'your_struct_name' is the variable name in MATLAB)
    groundtruth = data[varName][0]
    names = groundtruth.dtype.names
    # print(f"Groundtruth: {groundtruth}")
    #print(f"Names: {names}\nLength: {len(groundtruth)}\nFirst: {groundtruth[0]}")

    # Convert the structured NumPy array to a Python dictionary
    # This often involves iterating through the fields and handling nested structures
    python_dict = {name: [] for name in groundtruth.dtype.names}

    #print(f"Python Dict: {python_dict}")

    for x in groundtruth:
        for i in range(len(names)):
            python_dict[names[i]].append(np.squeeze(x[i]))

    #df = pd.DataFrame({"Fields": python_dict.keys(), "Values": python_dict.values()})
    df = pd.DataFrame(python_dict)

    df.to_csv(f"{varName}.csv")

    print(df.head(10))

    #for name in names: print(f"Updated {name}: {python_dict[name][0]}")

groundtruthConv()

allsplitConv()