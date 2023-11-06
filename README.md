# thesis_new
a new version of the thesis repo that actually works 

need to redo dataset1.  - or not, there seems to be a zero crossing occuring, which is apparant in both weight and non weight versions. 
need to plot against angle. 


pmr.py reco radial_cine_2d -i data.twix -o aw1_rieseling_admm_tgv_5e-2.nii -g ../scan_data_11092023_2DRadial_60bpm_NW.dat -aw 1 -os 5 -d neg -zf 2 -v -e riesling-admm -ea "--sense-fov=384.0,384.0,3.0 --fov=384.0,384.0,3.0 --tgv=5e-2 --max-outer-its=4"

