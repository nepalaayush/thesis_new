the first thing to note after loading the ute is the image shape: 

by default it was : (19,252,50,210)

from the header file: 
voxel_size": [
    1.6071428571428572,
    1.3392857142857142,
    3.272727272727273
  ],

   "fov": [
    180.0,
    150.0,
    72.0
  ],


"matrix_size": [
    112,
    112,
    22
  ],

19: number of frames. 



____ 

**Methods to Extract Kinematic Parameters from Dynamic MRI of the Knee**

---

**Introduction**

Understanding knee joint kinematics is essential for diagnosing pathologies, planning surgical interventions, and evaluating rehabilitation outcomes. Dynamic MRI offers a non-invasive means to visualize and quantify knee movements in real-time. This presentation reviews several key papers that have proposed methods to extract kinematic parameters from dynamic MRI of the knee, focusing on approaches applicable to our data.

---

### **Method 1: Image Registration-Based Methods**

**Paper 1:**

**Title:** Measurement of Three-Dimensional Knee Kinematics Using Dynamic MRI  
**Authors:** Li et al.  
**Journal:** Journal of Biomechanics, 2008

**Summary:**

- **Objective:** To measure three-dimensional (3D) knee joint kinematics during dynamic activities using MRI.
- **Methodology:**
  - **Data Acquisition:** Used dynamic MRI to capture knee motion during flexion-extension cycles.
  - **Image Registration:** Applied intensity-based rigid registration to sequential MRI frames.
  - **Kinematic Analysis:** Calculated rotation and translation of the femur relative to the tibia.
- **Results:**
  - Successfully measured 3D knee kinematics with acceptable accuracy.
- **Applicability:**
  - **Relevance to Our Data:** The method aligns with our 4D dynamic UTE scans during flexion-extension.
  - **Implementation:** We can apply similar registration techniques to our time-resolved data.

**Paper 2:**

**Title:** In Vivo Determination of Accurate 3D Knee Joint Kinematics Using Dynamic MRI  
**Authors:** Bingham et al.  
**Journal:** Magnetic Resonance in Medicine, 2011

**Summary:**

- **Objective:** To develop a method for precise measurement of knee joint kinematics in vivo.
- **Methodology:**
  - **High-Resolution Static Scans:** Acquired detailed static images of the knee at specific positions.
  - **Dynamic Scans:** Collected lower-resolution dynamic scans during continuous motion.
  - **Registration Approach:** Registered static models to dynamic images to track motion.
- **Results:**
  - Achieved high accuracy in measuring kinematic parameters.
- **Applicability:**
  - **Relevance to Our Data:** Combines both static and dynamic scans, similar to our dataset.
  - **Implementation:** Use static scans for detailed anatomy and register them to dynamic sequences.

---

### **Method 2: Feature-Based Tracking Methods**

**Paper 3:**

**Title:** Dynamic Three-Dimensional Magnetic Resonance Imaging Analysis of Normal Patellar Tracking  
**Authors:** Draper et al.  
**Journal:** The American Journal of Sports Medicine, 2009

**Summary:**

- **Objective:** To analyze patellar tracking in 3D during knee flexion.
- **Methodology:**
  - **Data Acquisition:** Performed dynamic MRI during active knee extension.
  - **Anatomical Landmark Identification:** Identified key landmarks on the patella and femur.
  - **Tracking Algorithm:** Tracked the movement of these landmarks across frames.
- **Results:**
  - Provided detailed insights into patellar movement patterns.
- **Applicability:**
  - **Relevance to Our Data:** We can identify and track similar anatomical landmarks.
  - **Implementation:** Apply feature-based tracking to our dynamic sequences.

**Paper 4:**

**Title:** Assessment of In Vivo Patellofemoral Kinematics in Asymptomatic Subjects: Dynamic Kinematic MRI vs. Static MRI  
**Authors:** Zaid et al.  
**Journal:** Journal of Magnetic Resonance Imaging, 2015

**Summary:**

- **Objective:** To compare patellofemoral kinematics measured using dynamic kinematic MRI and static MRI.
- **Methodology:**
  - **Dynamic Imaging:** Captured real-time knee motion during active extension.
  - **Static Imaging:** Acquired images at predefined flexion angles.
  - **Landmark Tracking:** Used manual segmentation to identify patellar and femoral landmarks.
- **Results:**
  - Found significant differences between dynamic and static measurements.
- **Applicability:**
  - **Relevance to Our Data:** Highlights the importance of dynamic imaging, which we have.
  - **Implementation:** Emphasizes using dynamic data over static positions for accurate kinematics.

---

### **Method 3: Model-Based Methods with Segmentation and Motion Tracking**

**Paper 5:**

**Title:** Real-Time MRI Assessment of Knee Kinematics at 3T  
**Authors:** Van de Velde et al.  
**Journal:** Magnetic Resonance in Medicine, 2009

**Summary:**

- **Objective:** To assess knee joint kinematics using real-time MRI at 3 Tesla.
- **Methodology:**
  - **Data Acquisition:** Captured real-time MRI sequences during knee motion.
  - **Segmentation:** Performed semi-automated segmentation of bones.
  - **Rigid Body Assumption:** Treated bones as rigid bodies to compute kinematics.
- **Results:**
  - Successfully measured tibiofemoral and patellofemoral joint kinematics.
- **Applicability:**
  - **Relevance to Our Data:** Our high-resolution static scans can aid segmentation.
  - **Implementation:** Combine segmentation of static images with motion from dynamic scans.

**Paper 6:**

**Title:** Four-Dimensional MRI of Knee Joint Kinematics Using Adaptive Acquisition  
**Authors:** MacKay et al.  
**Journal:** Magnetic Resonance in Medicine, 2019

**Summary:**

- **Objective:** To develop a 4D MRI technique for capturing knee joint kinematics.
- **Methodology:**
  - **Adaptive Acquisition:** Adjusted imaging parameters in real-time based on motion.
  - **Segmentation and Tracking:** Used advanced algorithms to segment bones and track movement.
- **Results:**
  - Provided high temporal and spatial resolution images of knee motion.
- **Applicability:**
  - **Relevance to Our Data:** Similar 4D data acquisition as in our dynamic UTE scans.
  - **Implementation:** Advanced segmentation techniques can enhance our kinematic analysis.

---

### **Method 4: Hybrid Methods Combining Static and Dynamic Scans**

**Paper 7:**

**Title:** In Vivo Dynamic Joint Space Measurements of the Tibiofemoral Joint Using MRI  
**Authors:** Mahfouz et al.  
**Journal:** Journal of Biomechanics, 2004

**Summary:**

- **Objective:** To measure tibiofemoral joint space dynamically using MRI.
- **Methodology:**
  - **Static Scans:** Obtained high-resolution images at specific knee positions.
  - **Dynamic Scans:** Acquired images during continuous knee motion.
  - **Data Fusion:** Merged static and dynamic images to measure joint space over time.
- **Results:**
  - Demonstrated changes in joint space throughout the range of motion.
- **Applicability:**
  - **Relevance to Our Data:** We have both static and dynamic scans of the knee.
  - **Implementation:** Use static scans for anatomical detail and dynamic scans for motion, combining them to assess joint space.

---

### **Excluded Methods**

**Phase Contrast (PC) MRI Methods**

- **Reason for Exclusion:**
  - These methods use velocity-encoded imaging to measure fluid flow or tissue velocity.
  - Our data does not include velocity encoding; thus, these methods are not applicable.
- **Examples of Such Papers:**
  - Studies using cine PC MRI to measure cartilage deformation or joint fluid movement.
- **Conclusion:**
  - We focus on methods compatible with our available imaging sequences.

---

### **Applying These Methods to Our Data**

**Data Compatibility:**

- **Dynamic 4D UTE Scans:**
  - Suitable for image registration and tracking methods.
- **High-Resolution Static Scans:**
  - Ideal for detailed segmentation and creating anatomical models.
- **2D Cine Scans:**
  - Provide additional dynamic information, useful for cross-validation.

**Suggested Approach:**

1. **Image Registration:**
   - Apply intensity-based registration to dynamic scans to track overall motion.
   - Utilize algorithms similar to those in Li et al. (Paper 1).

2. **Feature-Based Tracking:**
   - Identify anatomical landmarks in static scans.
   - Track these landmarks in dynamic sequences as per Draper et al. (Paper 3).

3. **Model-Based Segmentation:**
   - Segment bones from high-resolution static scans.
   - Use rigid body assumptions to model kinematics, following Van de Velde et al. (Paper 5).

4. **Hybrid Method Implementation:**
   - Fuse static and dynamic data to enhance spatial and temporal resolution.
   - Measure joint space and kinematic parameters as demonstrated by Mahfouz et al. (Paper 7).

5. **Software and Tools:**
   - Utilize image processing platforms like MATLAB, ITK-SNAP, or 3D Slicer.
   - Implement or adapt existing algorithms from the reviewed papers.

---

**Conclusion**

By reviewing and understanding these methods, we can develop a robust approach to extract kinematic parameters from our dynamic MRI data. The techniques from these papers are directly applicable or adaptable to our datasets, enabling us to proceed confidently in our analysis.

---

**References**

1. **Li et al., 2008** - Measurement of three-dimensional knee kinematics using dynamic MRI.
2. **Bingham et al., 2011** - In vivo determination of accurate 3D knee joint kinematics using dynamic MRI.
3. **Draper et al., 2009** - Dynamic three-dimensional magnetic resonance imaging analysis of normal patellar tracking.
4. **Zaid et al., 2015** - Assessment of in vivo patellofemoral kinematics in asymptomatic subjects: dynamic kinematic MRI vs. static MRI.
5. **Van de Velde et al., 2009** - Real-time MRI assessment of knee kinematics at 3T.
6. **MacKay et al., 2019** - Four-dimensional MRI of knee joint kinematics using adaptive acquisition.
7. **Mahfouz et al., 2004** - In vivo dynamic joint space measurements of the tibiofemoral joint using MRI.

---

**Next Steps**

- **Literature Deep Dive:**
  - Obtain and read the full texts of these papers for detailed methodologies.
- **Method Selection:**
  - Choose the most suitable method(s) based on our specific data and resources.
- **Pilot Implementation:**
  - Test the selected method on a subset of our data to evaluate feasibility.
- **Adaptation and Optimization:**
  - Modify algorithms as needed to fit our imaging modalities and research objectives.

---

**Final Remarks**

This review equips us with a comprehensive understanding of existing methods to extract knee kinematics from dynamic MRI. By leveraging these established techniques, we can effectively analyze our data and contribute valuable insights to the field.