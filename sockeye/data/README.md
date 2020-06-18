# ENT-DESC dataset


## Data Preparation
To prepare our dataset, we first use [Nayuki’s implementation](https://www.nayuki.io/page/computing-wikipedias-internal-pageranks) to calculate the PageRank score for more than 9.9 million Wikipages. We then extract the categories from Wikidata for the top 100k highest scored pages and manually select 90 categories out of the top 200 most frequent ones as the seed categories. The domains of the categories mainly include humans, events, locations and organizations. The entities from these categories are collected as our candidate set of main entities. We further process their associated Wikipedia pages for collecting the first paragraphs and entities with hyperlink as topic-related entities. We then search Wikidata to gather neighbours of the main entities and 1-hop/2-hop paths between main entities and their associated topic-related entities, which finally results in a dataset consisting of more than 110k entity-text pairs with 3 million triples in the KG.

## Data Preprocessing
We need to convert the dataset into multi graphs for training. For details please refer to the [paper](https://arxiv.org/pdf/2004.14813.pdf).

The preprocessed dataset is saved in `./ENT-DESC\ dataset`. Note, due to the dataset size, training files are saved as `.zip`.

## Citation
```
@article{cheng2020knowledge,
  title={Knowledge Graph Empowered Entity Description Generation},
  author={Cheng, Liying and Zhang, Yan and Wu, Dekun and Jie, Zhanming and Bing, Lidong and Lu, Wei and Si, Luo},
  journal={arXiv preprint arXiv:2004.14813},
  year={2020}
}
```

