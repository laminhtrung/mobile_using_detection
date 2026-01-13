python test.py \
  --weights /root/trunglm8/mobile_phone_detection/runs/train/ver_26_9/weights/best.pt \
  --data /root/trunglm8/mobile_phone_detection/MUID-IITR/data_29_9/data.yaml \
  --imgsz 640 --batch 32 --device 0 \
  --iou 0.6 --max_det 300 \
  --project runs/test --name test_30_9 \
  --plots 
