import numpy as np

def calc_ece(y,y_prim,bin_count):
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
    return np.sum(bin_eces*bin_counts)/y.shape[0]

def nll(y_prim,y):
    loss = -np.mean(y * np.log(y_prim + 1e-8))
    return loss
