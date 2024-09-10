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

radial_cine_2d parameters:
  -g GATING, --gating GATING
                        file with gating signal
  -ngn, --nogatingnorm  Do not normalize gating signal to maximum
  -e {grid,riesling-lsq,riesling-rlsq}, --engine {grid,riesling-lsq,riesling-rlsq}
                        reconstruction engine to use
  -ea ENGINEARGS, --engineargs ENGINEARGS
                        additional arguments for reconstruction engine
  -os OFFSETSTART, --offsetstart OFFSETSTART
                        Offset to start angle
  -oe OFFSETEND, --offsetend OFFSETEND
                        Offset to end angle
  -rs REPSTART, --repstart REPSTART
                        Start repetition
  -re REPEND, --repend REPEND
                        End repetition
  -zf ZEROFILLING, --zerofilling ZEROFILLING
                        Zero filling factor
  -cf FOVFACTOR, --fovfactor FOVFACTOR
                        FoV cut factor
  -aw ANGLEWINDOW, --anglewindow ANGLEWINDOW
                        Angle window width
  -ai ANGLEINCREMENT, --angleincrement ANGLEINCREMENT
                        Angle increment between frames
  -d {neg,pos,any}, --direction {neg,pos,any}
                        Use spokes where gating slope is positive, negative or do not gate by slope
  -sf {qff,fff,ddd}, --sensorformat {qff,fff,ddd}
                        sensor file format to use (ddd=double, fff=float, qff=timefloat)



for 3d: 

pmr.py reco radial_cine_3d -i meas_MID00116_FID309381_MK_UTE_W_CINE_60bpm_S96.dat -o CON_07_3d_riesling_pos.nii -g scan_data_CON07_3D_W.dat -e riesling-rlsq -ea "--sense-fov=270.0,225.0,108.0 --osamp 3 --fov=270.0,225.0,108.0 --tgv=5e-4 --max-outer-its=10" -os 2 -oe 2 -ds 1 -aw 2 -ai 2 -d pos -cf 1.5 -fs 1.5 -sf qff -v

for2d: 
pmr.py reco radial_cine_2d -i meas_MID00140_FID309405_MK_Radial_NW_CINE_30bpm_CGA.dat -o CON_18_2d_NW_pos.nii -g scan_data_CON18_2D_NW.dat -sf qff  -aw 2 -ai 5 -d pos -zf 2  -v -cf 1 -e riesling-rlsq -ea "--tgv=5e-3 --osamp 3 --sense-fov=384.0,384.0,3.0 --fov=384.0,384.0,3.0 --max-outer-its=10"