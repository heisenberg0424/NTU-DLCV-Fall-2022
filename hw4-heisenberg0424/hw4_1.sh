#!/bin/bash
python3 DirectVoxGO_TEST/run.py --jsonpath $1 --config DirectVoxGO_TEST/configs/nerf/hotdog.py --render_only --render_test --outputpath $2
# TODO - run your inference Python3 code