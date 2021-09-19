# PARENT: Table to Text Evaluation

This folder contains scripts for computing the PARENT metric for table-to-text
evaluation, which is described in the following paper:

[Handling Divergent Reference Texts when Evaluating Table-to-Text Generation](https://arxiv.org/abs/1906.01081)\
Bhuwan Dhingra, Manaal Faruqui, Ankur Parikh, Ming-Wei Chang, Dipanjan Das, William W. Cohen\
ACL, 2019

## Computing PARENT

PARENT evaluates the generated text against both references and the table
itself. To compute PARENT on the generations in a file `<generation_file>`:

```
python -m table_text_eval \
  --references <reference_file> \
  --generations <generation_file> \
  --tables <table_file>
```

Example:

```
python -m table_text_eval \
  --references test_surface.pp.txt \
  --generations tg_mgcn_avg-2.txt \
  --tables test_table.txt
```

Where the tables and references are in the `<table_file>` and `<reference_file>`,
respectively. See the `table_text_eval.py` script for more details on the
format for these files.

`test_table.txt` is a processed table file using `triple2table.py`.


## Citation

```
@inproceedings{dhingra2019handling,
  title={Handling divergent reference texts in table-to-text generation},
  author={Dhingra, Bhuwan and Faruqui, Manaal and Parikh, Ankur and Chang, Ming-Wei and Das, Dipanjan and Cohen, William W},
  booktitle={Proc. of ACL},
  year={2019}
}
```
