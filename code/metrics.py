import numpy as np

def calc_ece2(y,y_prim,bin_count):
    bins,step=np.linspace(0,1,bin_count,endpoint=False,retstep=True)
    y_idx=np.argmax(y,1)
    y_prim_idx=np.argmax(y_prim,1)
    conf=np.max(y_prim,1)
    correct=np.where(y_idx==y_prim_idx,1,0)
    bin_idxs=np.digitize(conf,bins)
    bin_eces=np.zeros_like(bins)
    bin_counts=np.zeros_like(bins)#I think the metric is more stable if results were weighed by bin size
    for i, bin in enumerate(bins):
        in_bin=bin_idxs==i+1
        in_bin_confs=conf[in_bin]
        in_bin_correct=correct[in_bin]
        bin_counts[i]=np.count_nonzero(in_bin)
        if(bin_counts[i]>0):
            bin_eces[i]=np.abs(np.mean(in_bin_confs)-np.mean(in_bin_correct))
        else:
            bin_eces[i]=0
    return np.mean(bin_eces)

def calc_ece(y,y_prim,bin_count):
    #Based from Uncertainty-Quantification-and-Deep-Ensemble\utils\uqutils.py , Rahul Rahaman, Alexandre H. Thiery
    eces=[]
    total_samples=y.shape[0]
    for clas in range(y.shape[1]):
        y_class=y[:,clas]
        y_prim_class=y_prim[:,clas]
        bins = (y_prim_class*bin_count).astype(int)
        ece_total = np.array([np.sum(bins == i) for i in range(bin_count+1)])
        ece_correct = np.array([np.sum((bins == i)*y_class) for i in range(bin_count+1)])
        acc = np.array([ece_correct[i]/ece_total[i] if ece_total[i] > 0 else -1 for i in range(bin_count+1)])
        conf = np.array([np.mean(y_prim_class[bins == i]) if ece_total[i] > 0 else 0 for i in range(bin_count+1)])
        deviation = np.sum([abs(acc[i] - conf[i])*ece_total[i] if acc[i] >= 0 else 0 for i in range(bin_count+1)])
        eces.append(deviation)
    return np.mean(eces)/total_samples
#print(calc_ece(np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0]]),np.array([[0.1,0.3,0.5],[0.2,0.4,0.4],[0.1,0,0.9],[0.2,0.3,0.4]]),bin_count=10))
def nll(y_prim,y):
    loss = -np.mean(y * np.log(y_prim + 1e-8))
    return loss
