## Code Reference
[DCGCN](https://github.com/Cartus/DCGCN)

## Dependencies
The model requires:
- Python3
- [MXNet 1.3.0](https://github.com/apache/incubator-mxnet/tree/1.3.0)
- [Sockeye 1.18.56](https://github.com/awslabs/sockeye)
- CUDA

## Installation
#### GPU

If you want to run sockeye on a GPU you need to make sure your version of Apache MXNet
Incubating contains the GPU bindings. Depending on your version of CUDA you can do this by running the following:

```bash
> pip install -r requirements/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install .
```
where `${CUDA_VERSION}` can be `75` (7.5), `80` (8.0), `90` (9.0), `91` (9.1), or `92` (9.2).

## ENT-DESC Dataset

The preprocessed ENT-DESC dataset is saved in `./sockeye/data`. For more details regarding the data preparation step, please refer to [ENT-DESC](https://github.com/LiyingCheng95/EntityDescriptionGeneration/tree/master/sockeye/data).

Before that, we need to convert the raw dataset into multi graphs for training. For details please refer to the [paper](https://arxiv.org/pdf/2004.14813.pdf).

## Training

To train the DCGCN model, run:

```
./train.sh
```

Model checkpoints and logs will be saved to `./sockeye/model`.

## Decoding

When we finish the training, we can use the trained model to decode on the test set, run:

```
./decode.sh
```

This will use the last checkpoint by default. Use `--checkpoints` to specify a model checkpoint file.

## Evaluation

For BLEU score evaluation, run:

```
python3 -m sockeye.evaluate -r sockeye/data/ENT-DESC\ dataset/test_surface.pp.txt  -i sockeye/data/ENT-DESC\ dataset/test.snt.out
```

## Citation
```
@article{cheng2020knowledge,
  title={Knowledge Graph Empowered Entity Description Generation},
  author={Cheng, Liying and Zhang, Yan and Wu, Dekun and Jie, Zhanming and Bing, Lidong and Lu, Wei and Si, Luo},
  journal={arXiv preprint arXiv:2004.14813},
  year={2020}
}
```

