#! /bin/bash
python scripts/extract_coco.py
cat ./data/val.txt | head -n  5000 > ./data/test.txt
python core/convert_tfrecord.py --dataset_txt ./data/test.txt  --tfrecord_path_prefix /home/qing/ngs/tensorflow-yolov3/tfrecords/test
