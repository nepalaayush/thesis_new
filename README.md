# thesis_new
a new version of the thesis repo that actually works 

need to redo dataset1.  - or not, there seems to be a zero crossing occuring, which is apparant in both weight and non weight versions. 
need to plot against angle. 


pmr.py reco radial_cine_2d -i data.twix -o MM_NW_aw2_tgv_5e-2_pos.nii -g ../scan_data_081223_2DRadial_MM_NW_30bpm.dat -aw 2 -os 5 -d pos -zf 2 -v -e riesling-admm -ea "--sense-fov=384.0,384.0,3.0 --fov=384.0,384.0,3.0 --tgv=5e-2 --max-outer-its=4"

with angle increment: 

pmr.py reco radial_cine_2d -i data.twix -o US_W_ai2_tgv_5e-2_pos.nii -g ../scan_data_11092023_2DRadial_60bpm_W.dat -aw 2 -os 5 -ai 2 -d pos -zf 2 -v -e riesling-admm -ea "--sense-fov=384.0,384.0,3.0 --fov=384.0,384.0,3.0 --tgv=5e-2 --max-outer-its=4"

with repetition:   
pmr.py reco radial_cine_2d -i data.twix -o MM_W_ai2_tgv_5e-2_neg_r0_r20.nii -g ../scan_data_MM_W_260124_30bpm.dat -aw 2 -os 5 -ai 2 -d neg -zf 2 -v -e riesling-admm -ea "--sense-fov=384.0,384.0,3.0 --fov=384.0,384.0,3.0 --tgv=5e-2 --max-outer-its=4" -rs 0 -re 20


for increasing fov: 
pmr.py reco radial_cine_2d -i data.twix -o NB_NW_ai2_tgv_5e-3_pos.nii -g ../scan_data_NB_NW_190124_25bpm_range.dat -aw 2 -os 5 -ai 2 -d pos -zf 2 -v -e riesling-admm -ea "--sense-fov=576.0,576.0,3.0 --fov=576.0,576.0,3.0 --tgv=5e-3 --osamp=3 --max-outer-its=4"


____   
without admm:   
pmr.py reco radial_cine_2d -i data.twix -o MM_NW_aw2_pos_base.nii -g ../scan_data_081223_2DRadial_MM_NW_30bpm.dat -aw 2 -os 5 -d pos -zf 2 -v -ngn

for admm and slope 
pmr.py reco radial_cine_2d -i data.twix -o MM_NW_aw2_tgv_5e-2_pos.nii -g ../scan_data_081223_2DRadial_MM_NW_30bpm.dat -aw 2 -ai 2 -d pos -zf 2 -ngn-  -v -e riesling-admm -ea "--sense-fov=384.0,384.0,3.0 --fov=384.0,384.0,3.0 --tgv=5e-2 --max-outer-its=4"


____________________

for obtaining the kspace : 
