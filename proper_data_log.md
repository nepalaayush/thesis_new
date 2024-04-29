**Subjects: MM **

datasets from 26.01 and before, call it 'old'

for NW: 

TO DO:  

1. reco using 2 angle increments. 

2. do fem segmentation first

for W: 

1. reco using the partial repetion and see if it gets better and obtain the data 

Results: 
NW: indeed, there is the cumulative deviation after a certain angle. 

___________________________________________________________________

For data acquired on 26.01 

NW and W both are decent datasets 

![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-03-18-12-38-image.png)

need to redo the NW part to explain this final frame weirdness. 

_____

For the subject MK: 

have two datasets one for left leg and one for right leg 

For left leg: 

![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-04-11-12-19-image.png)

Not bad but still a bit of deviation. 

- the femur looks fine no through plane motion here 

TO DO: have to do the proper regardless, so need to do again, no ref frame 

re calculating the transformation matrices, twice, the result is a bit deviation near the plateau:
![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-04-11-35-48-image.png)

![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-04-11-36-16-image.png)

This was improved, the mae, by tweaking the curve of the first frame. still not a satisfactory result. 

_______________________________________________________________________________________

For MK data taken on 02.02: 

Starting with NW: 

- [x] final label tib 

- [x] t matrix 

- [x] ref frame 

- [x] segmented shape 

- [x] info dict 

But the analysis part is messed up because the orientation is the opposite. 

![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-09-15-54-07-image.png)

very nice looking plot at least for the centroid. worth looking into. 

MK_W_02.02: (just reverse the order and do it once more .)

- [ ] final label tib

- [ ] t matrix

- [ ] ref frame

- [ ] segmented shape

- [ ] info dict

- [ ] final label fem
  
  - [ ] t matrix
  
  - [ ] ref frame
  
  - [ ] segmented shape
  
  - [ ] info dict

super weird stuff. so, the angles do not make sense AT all, there is total deviation. . but for the one i did previously, there is NO deviation. one of the best datasets. so why? i even removed the first reference frame and then did it again, but am facing the same problem. 

- All i want to do now, is to just obtain one single decent loaded and unloaded case. so maybe simply cheat the system and do it as is .using the other ref matrices. 

here is what is new: 
it absolutely matters if we start from the first frame or the last. it really doesnt matter if it is extended or flexed. the results can vastly vary, even though the final label is the same. this seems super odd.. but it has to do with the cumulation of errors here. 

For AN NW: 
![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-09-18-50-20-image.png)

super bad angles, this was going from first frame to last, where first was fully extended 

![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-09-18-50-55-image.png)

now this is from last to first. here, the first two are actually behaving well. even though from third, we would expect some kind of deviation, but that simply does not happen. 

Now, here is the most radical idea: complete manual segmentation, no edge image at ALL! 

: ![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-09-19-03-40-image.png)

slightly better than the edge, but still pretty bad. 

___ ___ ___ ___ ___ ___ 

- Mention dr. Krämer that the extension for the subjects should be forced to go all the way. 

____ __ ___ __ __ __ 



some pmr.py info stuff taken directly from the raw file: 

      INFO     MRIArray(
2024-04-29 11:44:57,784 pymri        INFO       shape = (352, 276, 16, 100)
2024-04-29 11:44:57,784 pymri        INFO       dims = ('read', 'line', 'channel', 'repetition')
2024-04-29 11:44:57,784 pymri        INFO       type = complex64
2024-04-29 11:44:57,784 pymri        INFO       header = 13 items
2024-04-29 11:44:57,784 pymri        INFO     )
2024-04-29 11:44:57,784 pymri        INFO     MRIHeader(
2024-04-29 11:44:57,784 pymri        INFO       acquisition_duration: 160
2024-04-29 11:44:57,784 pymri        INFO       acquisition_type: 2D
2024-04-29 11:44:57,784 pymri        INFO       dwell_time: [0.0024]
2024-04-29 11:44:57,784 pymri        INFO       echo_time: [2.5100000000000002]
2024-04-29 11:44:57,784 pymri        INFO       field_strength: 2.89362
2024-04-29 11:44:57,784 pymri        INFO       flip_angle: 8.0
2024-04-29 11:44:57,784 pymri        INFO       fov: [192.0, 192.0, 3.0]
2024-04-29 11:44:57,784 pymri        INFO       frequency: 123248564
2024-04-29 11:44:57,784 pymri        INFO       inversion_time: 150000
2024-04-29 11:44:57,784 pymri        INFO       matrix_size: [176, 176, 1]
2024-04-29 11:44:57,785 pymri        INFO       repetition_time: 5.8
2024-04-29 11:44:57,785 pymri        INFO       scan_options: recoInfo_20240405_1033_770
2024-04-29 11:44:57,785 pymri        INFO       scanner_name: MAGNETOM Prisma
2024-04-29 11:44:57,785 pymri        INFO     )


