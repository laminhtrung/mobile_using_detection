python3 pipeline_phone_use.py \
  --mode video \
  --source "rtsp://user:pass@ip:554/Streaming/Channels/101" \
  --person-weights /root/trunglm8/mobile_phone_detection/yolo11n.pt \
  --phone-weights  /root/trunglm8/mobile_phone_detection/runs/train/support_autolabel_version/weights/best.pt \
  --device 0 \
  --draw-phone-box \
  --save-video /root/trunglm8/mobile_phone_detection/runs/video/cam1_annotated.mp4 \
  --show \
  --outdir /root/trunglm8/mobile_phone_detection/runs/video
