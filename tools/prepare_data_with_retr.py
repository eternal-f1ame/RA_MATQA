import numpy as np
import json

if __name__ == '__main__':
    with open('nearest_neighbors.npy', 'rb') as f:
        retrieved_neighbors = np.load(f, allow_pickle=True)[()]
    
    splits = ['train', 'val', 'test']

    train_indices = retrieved_neighbors['train']
    val_indices = retrieved_neighbors['val']
    test_indices = retrieved_neighbors['test']

    pairs_data = {}

    for split in splits:
        with open(f'data/MathQA_bert_token_{split}.json') as f:
            pairs_data[split] = json.load(f)['pairs']
    

    for split in splits:
        for i, ind in enumerate(eval(f'{split}_indices')):
            pair = pairs_data['train'][ind]

            pairs_data[split][i]['tokens'].extend(pair['tokens'][1:]) 
            pairs_data[split][i]['expression'].extend(pair['expression']) 
            pairs_data[split][i]['nums'].extend(pair['nums']) 
            pairs_data[split][i]['num_pos'].extend(pair['num_pos'])
    

    for split, pairs in pairs_data.items():
        with open(f'data/MathQA_bert_token_{split}.json') as f:
            data = json.load(f)
        data['pairs'] = pairs

        with open(f'ret_aug_data/mathqa_{split}.json', 'w') as f:
            json.dump(data, f, indent=4)


