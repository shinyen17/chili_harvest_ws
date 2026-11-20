Put your trained YOLO weights here, then rebuild:

  chili_ripeness.pt   - ripeness OBJECT DETECTION model (ripe / semi-ripe / unripe)
  chili_stem_seg.pt   - INSTANCE SEGMENTATION model (chili, stem)

Use best.pt from each training run (runs/detect/.../weights/best.pt and
runs/segment/.../weights/best.pt), renamed as above. Different names are fine
if you change ripeness_model / seg_model in config/chili_params.yaml.
