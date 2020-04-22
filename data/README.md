# Prepare data

## COCO

### Download COCO captions and preprocess them

Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then do:

```bash
$ python scripts/prepro_labels_attributes.py --input_json data/dataset_coco.json --output_json data/cocotalk_attr.json --output_h5 data/cocotalk_attr
```

`prepro_labels_attributes.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk_attr.json` and discretized caption data are dumped into `data/cocotalk_label_attr.h5`.Note that we change this file for a little bit. The first 1000 words form the attribute vocabulary.


### Download Bottom-up features

Download pre-extracted feature from [link](https://github.com/peteanderson80/bottom-up-attention). You can either download adaptive one or fixed one.

For example:
```
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip

```

Then:

```bash
python scripts/make_bu_data.py --downloaded_feats where-you-place-the-updown-feature --output_dir data/cocobu
```

This will create `data/cocobu_fc_36`, `data/cocobu_att_36` and `data/cocobu_box_36`. We use the '36' feature for the sake of easier implementation of MAD.

### Prepare Addtional labels

We extract attribute labels, subsequent attribute labels, and the transition matrix from `data/cocotalk_label_attr.h5`.
```bash
python prepare_addtional_data.py
```
