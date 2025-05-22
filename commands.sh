# Calibration of images
python calibration.py /storage/janezk/databases/wool_data/01/L_front.png --output L_front_calibrated.jpg
python calibration.py /storage/janezk/databases/wool_data/01/L_left.png --output L_left_calibrated.jpg
python calibration.py /storage/janezk/databases/wool_data/01/L_right.png --output L_right_calibrated.jpg
python calibration.py /storage/janezk/databases/wool_data/01/L_top.png --output L_top_calibrated.jpg
python calibration.py /storage/janezk/databases/wool_data/01/L_sides.png --output L_sides_calibrated.jpg

# Optional fusion
python multi_light_fusion.py mean L_left_calibrated.jpg L_right_calibrated.jpg L_top_calibrated.jpg L_sides_calibrated.jpg L_front_calibrated.jpg --output L_mean_calibrated.jpg

# Cropping
python crop_wool.py L_mean_calibrated.jpg --output L_crop.jpg --border 130 80 70 80

# Edge detection
python canny_edges.py L_crop.jpg --scale 0.25 --mask_path L_canny.jpg --overlay_path L_overlay.jpg

# Histogram computation
python orientation_hist.py --image L_crop.jpg --edges L_canny.jpg --output L_hist.jpg --bin_size 3