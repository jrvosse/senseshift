import sys 
import collections
import pandas as pd
import numpy as np
import scipy.stats

sys.path.insert(0, "./histwords")
from representations.sequentialembedding import SequentialEmbedding

gold='datasets/HW/evaluationset_dataset0_manual.csv'
embeddings="embeddings/eng-all_sgns"

if __name__ == "__main__":
    df = pd.read_csv(gold)
    embeddings = SequentialEmbedding.load(embeddings, range(1800, 2000, 10))
    for index, row in df.iterrows():
        target = row['target']
        ref    = row['ref']
        gold   = int(row['gold'])
        offset = int(row['t'])
        time_sims = embeddings.get_time_sims(target, ref)
        t = collections.OrderedDict([])
        for y in range(offset, 2000, 10): t[y]=time_sims[y]
        rho, p = scipy.stats.spearmanr(t.keys(), t.values())
        if p < 0.05:
            if np.sign(rho) == np.sign(gold):
                print "significant and correct: {target:s}/{ref:s}, {rho:0.2f}, {p:0.2f}, {series:s}".format(target=target, ref=ref, rho=rho,p=p, series=str(t.items()))
            else:
                print "significant and incorrect: {target:s}/{ref:s}, {rho:0.2f}, {p:0.2f}, {series:s}".format(target=target, ref=ref, rho=rho,p=p, series=str(t.items()))
        else:
            if np.sign(rho) == np.sign(gold):
                print "insignificant and correct: {target:s}/{ref:s}, {rho:0.2f}, {p:0.2f}, {series:s}".format(target=target, ref=ref, rho=rho,p=p, series=str(t.items()))
            else:
                print "insignificant and incorrect: {target:s}/{ref:s}, {rho:0.2f}, {p:0.2f}, {series:s}".format(target=target, ref=ref, rho=rho,p=p, series=str(t.items()))
