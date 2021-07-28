# Diabetic Retinopathy Detection by Retinal segmentation with Region merging using CNN

![Header](/scr/Header.png)

**Diabetic retinopathy is a state of eye infirmity in which  damage  arises  due  to  diabetes  mellitus.  It  is  one  of  the  prominent  reasons  behind  blindness.  Most  of  new  cases  of  diabetic retinopathy can get reduced with proper treatment of eyes**



**Stages Of Diabetic Retinopathya**  

**Mild nonproliferative retinopathy**

During the early stage of the disease microaneurysms occures. microaneurysms are small areas where balloon-like swelling occurs in the blood vessels of the retina. It causes the fluid to leak into the retina.

**Moderate**

nonproliferative retinopathy.While the disease advances, the blood vessels begin to swell and distort. This completely affects their ability to transport blood. These conditions incite change in the appearance of the retina. 

**Severe**

nonproliferative retinopathy.During this state blood vessels gets completely blocked. This seizes blood supply to areas of the retina. These areas disguise growth factors and gives signal the retina to grow new blood vessels.

**Proliferative diabetic retinopathy (PDR)**

This is the highest stage of severity of diabetic retinopathy. At this stage, the growth factor triggers the retina to form new blood vessels. These new blood vessels are fragile and are likely to leak and bleed. This lead to the contraction of scar tissues which causes retinal detachment. Retinal detachment is a process of ripping away of the retina from the underlying tissue. The retinal detachment can cause permanent vision loss.

Reference: https://ieeexplore.ieee.org/document/8721315



#### Misc

Regional merging can be intergated in the data loader pipeline of CNN itself, but it will be taking time as it may repeat the merging on image twice or more. Rather merge the retinal segmenation on to image, so we can elimate the UNet part in the CNN pipeline



#### Reference:

- https://ieeexplore.ieee.org/document/8721315
- https://www.degruyter.com/document/doi/10.1515/bmt-2020-0089/html
- https://github.com/DeepTrial/Retina-VesselNet
- https://www.hindawi.com/journals/jhe/2017/4897258/
- https://www.researchgate.net/publication/332263231_Generation_of_binary_mask_of_retinal_fundus_image_using_bimodal_masking

