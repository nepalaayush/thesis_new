Subjects: MM 

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

-  the femur looks fine no through plane motion here 



TO DO: have to do the proper regardless, so need to do again, no ref frame 

re calculating the transformation matrices, twice, the result is a bit deviation near the plateau:
![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-04-11-35-48-image.png)

![](C:\Users\Aayush\AppData\Roaming\marktext\images\2024-02-04-11-36-16-image.png)

This was improved, the mae, by tweaking the curve of the first frame. still not a satisfactory result. 
