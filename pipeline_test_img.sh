python3 pipeline_test.py \
  --mode image \
  --source /root/trunglm8/mobile_phone_detection/crop_person_data/16_huy \
  --recursive \
  --person-weights /root/trunglm8/mobile_phone_detection/yolov9m.pt \
  --phone-weights  /root/trunglm8/mobile_phone_detection/runs/train/ver_24_9/weights/best.pt \
  --device 0 \
  --draw-phone-box --save-crops --export-json \
  --export-yolo-phone --yolo-phone-class 0 \
  --outdir /root/trunglm8/mobile_phone_detection/crop_person_data/16_huy
  