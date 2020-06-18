#!/usr/bin/env bash
python3 -m sockeye.translate -m sockeye/model \
        --edge-vocab sockeye/data/ENT-DESC\ dataset/edge_vocab.json < sockeye/data/ENT-DESC\ dataset/test.amrgrh \
        -o sockeye/data/ENT-DESC\ dataset/test.snt.out \
        --beam-size 10 \
        --checkpoints 200
