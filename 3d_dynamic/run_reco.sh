#!/bin/bash

# Run the first reconstruction
cd /data/projects/dfg-berlin-kneedynamics/data/Schulz^Helena/2024-09-13/28_MK_UTE_Static_Upper_HR
pmr.py reco radial_3d -id data.twix -o HS_UTE_Static_Upper_HR_riesling.nii -cf 1.5 -v -g girf_prisma.npy -lci -e riesling-rlsq -fs 1.5

# Run the second reconstruction
cd /data/projects/dfg-berlin-kneedynamics/data/Schulz^Helena/2024-09-13/29_MK_UTE_Static_Lower_HR
pmr.py reco radial_3d -id data.twix -o HS_UTE_Static_Lower_HR_riesling.nii -cf 1.5 -v -g girf_prisma.npy -lci -e riesling-rlsq -fs 1.5

echo "Both reconstructions completed."
