#!/bin/bash

python create_poisoned_set.py -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=mobilenetv2

python train_on_poisoned_set.py -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=mobilenetv2


python test_model.py -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=mobilenetv2


python test_stl10.py -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=mobilenetv2


python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=mobilenetv2




python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=mobilenetv2

python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=mobilenetv2


python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=mobilenetv2


python other_defense.py -defense=NC -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=mobilenetv2



