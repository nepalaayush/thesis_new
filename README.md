# thesis_new
a new version of the thesis repo that actually works 

need to redo dataset1.  - or not, there seems to be a zero crossing occuring, which is apparant in both weight and non weight versions. 
need to plot against angle. 


pmr.py reco radial_cine_2d -i data.twix -o MM_NW_aw2_tgv_5e-2_pos.nii -g ../scan_data_081223_2DRadial_MM_NW_30bpm.dat -aw 2 -os 5 -d pos -zf 2 -v -e riesling-admm -ea "--sense-fov=384.0,384.0,3.0 --fov=384.0,384.0,3.0 --tgv=5e-2 --max-outer-its=4"

