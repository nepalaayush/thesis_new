like NW, need to do proper reco for this one 

pmr.py reco radial_cine_2d -i data.twix -o AN_W_ai2_tgv_5e-2_pos_os3.nii -g ../scan_data_AN_W_020224_30bpm.dat -aw 2 -os 5 -ai 2 -d pos -zf 2 -v -e riesling-admm -ea "--sense-fov=576.0,576.0,3.0 --fov=576.0,576.0,3.0 --tgv=5e-2 --osamp=3 --max-outer-its=4"
2024-02-12 12:59:49,345 pymri        INFO     Loading data.twix
2024-02-12 12:59:49,345 pymri        INFO     Loading Siemens raw data from data.twix
2024-02-12 12:59:53,670 pymri        INFO     [0]: AdjCoilSens (31.81 MB)
2024-02-12 12:59:53,670 pymri        INFO     [1]: MK_Radial_W_CINE_30bpm_CGA (1207.15 MB) <---
2024-02-12 13:00:18,445 pymri        INFO     Loading recoInfo file from: data.recoInfo
2024-02-12 13:00:19,822 pymri        INFO     Number of detected triggers: 101
2024-02-12 13:00:19,823 pymri        INFO     Average time between triggers: 1600.8 +/- 0.4
2024-02-12 13:00:20,381 pymri        INFO     Using window width of 2.0 deg
2024-02-12 13:00:20,840 pymri        INFO     00: Reconstructing Frame using 653 spokes for angle 6 deg
2024-02-12 13:00:35,342 pymri        INFO     01: Reconstructing Frame using 646 spokes for angle 8 deg
2024-02-12 13:00:48,279 pymri        INFO     02: Reconstructing Frame using 789 spokes for angle 10 deg
2024-02-12 13:01:02,094 pymri        INFO     03: Reconstructing Frame using 837 spokes for angle 12 deg
2024-02-12 13:01:16,281 pymri        INFO     04: Reconstructing Frame using 800 spokes for angle 14 deg
2024-02-12 13:01:30,136 pymri        INFO     05: Reconstructing Frame using 984 spokes for angle 16 deg
2024-02-12 13:01:45,045 pymri        INFO     06: Reconstructing Frame using 1166 spokes for angle 18 deg
2024-02-12 13:02:00,899 pymri        INFO     07: Reconstructing Frame using 800 spokes for angle 20 deg
2024-02-12 13:02:14,687 pymri        INFO     08: Reconstructing Frame using 775 spokes for angle 22 deg
2024-02-12 13:02:28,218 pymri        INFO     09: Reconstructing Frame using 987 spokes for angle 24 deg
2024-02-12 13:02:43,036 pymri        INFO     10: Reconstructing Frame using 763 spokes for angle 26 deg
2024-02-12 13:02:56,519 pymri        INFO     11: Reconstructing Frame using 521 spokes for angle 28 deg
2024-02-12 13:03:08,404 pymri        INFO     12: Reconstructing Frame using 159 spokes for angle 30 deg
2024-02-12 13:03:17,579 pymri        INFO     Output data:MRIArray(
2024-02-12 13:03:17,579 pymri        INFO     	shape = (13, 792, 792)
2024-02-12 13:03:17,579 pymri        INFO     	dims = ('repetition', 'line', 'read')
2024-02-12 13:03:17,580 pymri        INFO     	type = float32
2024-02-12 13:03:17,580 pymri        INFO     	header = 6 items
2024-02-12 13:03:17,580 pymri        INFO     )
2024-02-12 13:03:17,580 pymri        INFO     Writing output to file: AN_W_ai2_tgv_5e-2_pos_os3.nii
