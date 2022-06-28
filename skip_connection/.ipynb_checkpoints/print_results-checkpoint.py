import pandas as pd
prune_ratio = [20, 36, 49, 59, 67, 74, 79, 83, 87, 89, 91, 93, 94, 95, 96, 97]

model = "resnet18"

for i in range(0, len(prune_ratio)):
    path1 = "../baseline_soft_activation/sparse_IMP/cifar100/"+ model + "/" +str(prune_ratio[i]) +"/log_scratch.txt"
    path2 = "./ls_swish_sparse_IMP/cifar100/"+ model + "/" +str(prune_ratio[i]) +"/log_scratch.txt"
    try:
        data = pd.read_csv(path1, sep = "\t")
        value1 = data.iloc[149, 4]
        
        data = pd.read_csv(path2, sep = "\t")
        value2 = data.iloc[149, 4]
        
        print("[{}] || Prune Ratio : {}\t LTH : {} || Skip-LTH : {}".format(model, prune_ratio[i], value1, value2))
    except:
        continue