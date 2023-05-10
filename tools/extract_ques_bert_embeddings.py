import os
import glob
import json
import torch
from transformers import AutoTokenizer, BertModel, BertTokenizer
import numpy as np
import tqdm


if __name__ == '__main__':
    files = glob.glob('data/MathQA_bert_*.json')
    print(files)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BertModel.from_pretrained('pretrained/MWP-BERT_en').to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('pretrained/MWP-BERT_en')
    tokenizer.add_special_tokens({"additional_special_tokens":["[num]"]})

    for f in files:
        split = f.split('_')[-1].split('.')[0]
        with open(f) as fl:
            data = json.load(fl)
    
        pairs = data['pairs']

        embeddings = []

        with torch.no_grad():
            for i, pair in enumerate(tqdm.tqdm(pairs)):
                tokens = pair['tokens']
                tokens_ = []

                for t in tokens:
                    if t.startswith('##'):
                        tokens_[-1] += t.replace('##', '')
                    else:
                        tokens_.append(t)
                
                inp = torch.tensor([tokenizer.encode(x, add_special_tokens=False)[0] for x in tokens_]).unsqueeze(0)

                out = model(inp.to(device))[0].squeeze().mean(0)

                embeddings.append(out.to('cpu').numpy())
        
        with open(f'embeddings/{split}.npy', 'wb') as ffl:
            np.save(ffl, embeddings)
        
        del embeddings

