#!/bin/bash

python -m LAM_3d_anymodel --model_name "RCAN" --window_size 24 --cube_no "001" \
       --h 50 --w 40 --d 40 --up_factor 1 --dataset_name "IXI" \
       --degradation_type "kspace_trunc" --input_type "2D"