# RA MATQA (Retrieval Augmented Math Question Answering)

We build our implementation based on the code from https://github.com/LZhenwen/MWP-BERT, thanks for their contribution!

## Network Training

**1. How to train the network on MathQA dataset:**
```
run mathqa.sh
```

## Weight Loading

To load the pre-trained MWP-BERT model, just change:

**mathqa.sh, Line 5:**
```
--bert_pretrain_path pretrained_models/bert-base-uncased \
```

Load the pre-trained model from your desired path.

## MWP-BERT weights

Please find at https://drive.google.com/drive/folders/1QC7b6dnUSbHLJQHJQNwecPNiQQoBFu8T?usp=sharing.

## Our Major Contribution

* The folder "tools" consists of code relevant to the Retrieval Augmented QA.
  * Code for dataset preparation is in the `prepare_data_with_retr.py`.
  * Code for fetching nearest neighbours on runtime is contained in `retrieve_nearest_neighbors.py` sequentially making use of `extract_ques_bert_embedding.py`.
  * Now run the `remove_watermark.py` file to see the results.

* The training and evaluation results are in `result/log.txt`
* For more comprehensive analysis and comparison, refer to [report](/documents/report.pdf).

### Contributors

> Aaditya Baranwal baranwal.1@iitj.ac.in ;  Github: [eternal-f1ame](https://github.com/aeternum) <br>
> Nakul Sharma sharma.86@iitj.ac.in ; Github: [thisis-nakul](https://github.com/thisis-nakul) <br>
> Saahil Bhavsar bhavsar.2@iitj.ac.in ; Github: [xander-watson](https://github.com/xander-watson) 
