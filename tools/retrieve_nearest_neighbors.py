import glob
import numpy  as np

if __name__ == '__main__':
    emb_files = glob.glob('embeddings/*.npy')
    embs = {}

    for fl in emb_files:
        with open(fl, 'rb') as f:
            print(fl)
            emb = np.load(f)
        embs[fl.split('.')[0].split('/')[-1]] = emb
    
    print(embs.keys())
    
    train_emb = embs['train']

    nearest_neighbor = {}
    for k in ['train', 'val', 'test']:
        e = embs[k]

        res = np.argsort(e @ train_emb.T)[:, 1]

        nearest_neighbor[k] = res.tolist()
    
    with open('nearest_neighbors.npy', 'wb') as f:
        np.save(f, nearest_neighbor)
