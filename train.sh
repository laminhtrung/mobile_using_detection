python3 train.py \
  --data /root/trunglm8/mobile_phone_detection/MUID-IITR/data_26_9/data.yaml \
  --hyp /root/trunglm8/mobile_phone_detection/config/hyp.yaml \
  --device 0,1\
  --optimizer SGD --epochs 170 --weights yolov9m.pt --batch 64\
  --name ver_26_9\
  --cos_lr


