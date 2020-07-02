# ENT-DESC dataset


## Data Preparation
To prepare our dataset, we first use [Nayuki’s implementation](https://www.nayuki.io/page/computing-wikipedias-internal-pageranks) to calculate the PageRank score for more than 9.9 million [Wikipages](https://dumps.wikimedia.org/enwiki/)(we use the dump released on July 1, 2019). We then extract the categories from Wikidata for the top 100k highest scored pages and manually select 90 categories out of the top 200 most frequent ones as the seed categories. The domains of the categories mainly include humans, events, locations and organizations. The detailed list of these categories and their QID can be found in the `categories_QID.txt`. The entities from these categories are collected as our candidate set of main entities. We further process their associated Wikipedia pages for collecting the first paragraphs and entities with hyperlink as topic-related entities. We then use the [Wikidata Query Service](https://query.wikidata.org/) to search Wikidata so as to gather neighbours of the main entities and 1-hop/2-hop paths between main entities and their associated topic-related entities. Note that we take bidirectional connection into account when collecting all the 1-hop/2-hop paths, which means that there are two types of 1-hop paths(i.e., A->B and B->A) and four types of 2-hop paths(i.e., A->C->B, A<-C->B, A->C<-B and A<-C<-B where A stands for the main entities, B stands for topic-related entities associated with A and C stands for the intermediate entities). This step finally results in a dataset consisting of more than 110k entity-text pairs with 3 million triples in the KG.

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


