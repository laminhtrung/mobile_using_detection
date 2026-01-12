python3 pipeline_test.py \
  --mode video \
  --source /root/trunglm8/mobile_phone_detection/video_test/test_video_30_9/test_7.mp4 \
  --person-weights /root/trunglm8/mobile_phone_detection/yolov9c.pt \
  --phone-weights  /root/trunglm8/mobile_phone_detection/runs/train/ver_26_9/weights/best.pt \
  --device 0 \
  --draw-phone-box --save-crops \
  --save-video /root/trunglm8/mobile_phone_detection/runs/video/test_7.mp4 \
  --outdir /root/trunglm8/mobile_phone_detection/runs/video
