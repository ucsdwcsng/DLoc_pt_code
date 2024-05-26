---
title: "WCSNG - Research"
layout: gridlay
excerpt: "WILD"
sitemap: true
permalink: /wild/
---

# Wireless Indoor Localization Dataset (WILD)
```
Authors: Roshan Ayyalasomayajula, Aditya Arun, Chenfeng Wu, Dinesh Bharadia
```

---

<div class="well">
 <center>
 <h4><A href="#documentation">Documentation</A>&emsp;&emsp;<A href="#downloads">Downloads</A>&emsp;&emsp;<a href="https://forms.gle/WWGymUFxhPWRc4zu7">LICENSE</a>&emsp;&emsp;<A href="#updates">Updates</A>&emsp;&emsp;<A href="#citation">CITATION</A></h4>
 </center>
</div>

---

## Updates ##
<i><font color="gray">December, 2022</font></i>
<p>2nd version of WILD dataset has been released through a <a href="https://www.kaggle.com/competitions/wildv2/data">Kaggle Competition</a>.</p>
<i><font color="gray">June 20, 2020</font></i>
<p>First release of the Location labelled WiFi CSI data and features data used in <a href="https://wcsng.ucsd.edu/dloc">DLoc</a>.</p>
<i><font color="gray">Jan, 2020</font></i>
<p><a href="https://wcsng.ucsd.edu/dloc">DLoc</a> has been accepted in Mobicom 2020.</p>

---

## Documentation ##

### Two Different Environments
<div class="col-sm-12 clearfix">

<div class="col-sm-6 clearfix">

<h4> 1. Complex High-multipath and NLOS environment (1500 sq. ft.) with 5 different setups in Jacobs Hall UCSD</h4>
<a href="{{ site.url }}{{ site.baseurl }}/images/respic/jacobs.png"><img src="{{ site.url }}{{ site.baseurl }}/images/respic/jacobs.png" width="100%" style="float: center" > </a>
- <img src="{{ site.url }}{{ site.baseurl }}/images/respic/jacobs_default.png" width="50%" style="float: center" >
<p>**jacobs_Jul28**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on July 28, 2019.</p>
- <img src="{{ site.url }}{{ site.baseurl }}/images/respic/jacobs_default.png" width="50%" style="float: center" >
<p>**jacobs_Jul28_2**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on July 28, 2019, one hour after **jacobs_Jul28**.</p>
- <img src="{{ site.url }}{{ site.baseurl }}/images/respic/jacobs_aug16_1.png" width="50%" style="float: center" >
<p>**jacobs_Aug16_1**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on August 16, 2019 with extra furniture placed randomly.</p>
- <img src="{{ site.url }}{{ site.baseurl }}/images/respic/jacobs_aug16_3.png" width="50%" style="float: center" >
<p>**jacobs_Aug16_3**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on August 16, 2019 with extra furniture placed randomly.</p>
- <img src="{{ site.url }}{{ site.baseurl }}/images/respic/jacobs_aug16_4_ref.png" width="50%" style="float: center" >
<p>**jacobs_Aug16_4_ref**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on August 16, 2019 with extra furniture placed randomly with an added reflector. (*a huge aluminium plated board*)</p>
</div>

<div class="col-sm-6 clearfix">
<h4> 2. Simple LOS based environment (500 sq. ft.) with 3 different setups in Atkison Hall UCSD </h4>
<a href="{{ site.url }}{{ site.baseurl }}/images/respic/atkinson.png"><img src="{{ site.url }}{{ site.baseurl }}/images/respic/atkinson.png" width="95%" style="float: center" > </a>
  - <img src="{{ site.url }}{{ site.baseurl }}/images/respic/atk_July22_1_ref.png" width="40%" style="float: center" >
  <p>**July16**: 8m X 5m setup with 3 APs in Atkinson hall ground floor for data collected on July 16, 2019.</p>
- <img src="{{ site.url }}{{ site.baseurl }}/images/respic/atk_July22_1_ref.png" width="40%" style="float: center" >
<p>**July18**: 8m X 5m setup with 3 APs in Atkinson hall ground floor for data collected on July 18, 2019.</p>
- <img src="{{ site.url }}{{ site.baseurl }}/images/respic/atk_July22_2_ref.png" width="40%" style="float: center" >
<p>**July22_2_ref**: 8m X 5m setup with 3 APs and 2 additonal reflectors (*a huge aluminium plated board*) placed in Atkinson hall ground floor for data collected on July 22, 2019.</p>
</div>
</div>

---

We provide both the CSI data for all the above setups and the post-prcessed features for running our DLoc network. **All the corresponding links can be found below.**

### Channels

The CSI data is named as **channels_<setup_name_from_above>.mat**. These MATLAB files are stored using **HDF5** file structure and contain the following variables:

- **channels**: *[ n_datapoints x n_frequency x n_ant X n_ap ]* 4D complex channel matrix.
- **RSSI**: *[ n_datapoints x n_ap ]* 2D recieved signal strenght matrix.
- **labels**: *[ n_datapoints x 2 ]* 2D XY labels.
- **opt**: various options specific for the data generated
	-*opt.freq* : *[n_frequencyx1]* 1D vector that describes the frequency of the subcarriers
	-*opt.lambda*: *[n_frequencyx1]* 1D vector that describes the wavelength of the subcarriers
	-*ant_sep*: antenna separation used on all of our APs
- **ap**: *n_ap* cell matrix. Each element corresposning to *[ n_ant x 2 ]* XY locations of the n_ant on each AP.
- **ap_aoa**: *[ n_ap x 1]* vectors that contains the rotation that needs to be added to the AoA measured at each AP (assumes that the AoA is measured about the normal to the AP's antenna array)
- **d1**: The sampled x-axis of the space under consideration
- **d2**: The sampled y-axis of the space under consideration

### Features

The 2D heatmap features data used in [DLoc](https://wcsng.ucsd.edu/dloc) is named as **features_<setup_name_from_above>.mat**. These MATLAB files are stored using **HDF5** file structure and contain the following variables:

- **features_with_offset**: *[ n_datapoints x n_ap x n_d1_points X n_d2_points ]* 4D feature matrix for n_ap **with offsets** in time
- **features_without_offset**: *[ n_datapoints x n_ap x n_d1_points X n_d2_points ]* 4D feature matrix for n_ap **without offsets** in time
- **labels_gaussian_2d**: *[ n_datapoints x n_d1_points X n_d2_points ]* 3D labels matrix that contisn the target images for the location network.
- **labels**: *[ n_datapoints x 2 ]* 2D XY labels.

---

## Downloads ##

<div class="col-sm-12 clearfix">

<div class="col-sm-6 clearfix">

### Channel Downloads:

Cumulative Downloads: [All channels](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EfERb1sUk65CjMrSIyD1b9kB0MFKf-d57gO3d7jRses6BQ?download=1)

Individual Downloads:
- [jacobs_Jul28](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/Ede931QqxmxFmHwiYz_H5dwBHVH8SnB02BjfAAWpD9FXXQ?download=1)
- [jacobs_Jul28_2](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/ESiVhglOHNNPh7h5IjGSz3ABzuzyDVI-XCzWBJFouu5IoA?download=1)
- [jacobs_Aug16_1](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/ER16mpDpebhMof2Gqd-hwoEB5koMPqkf7WKFbnGzsXaoRQ?download=1)
- [jacobs_Aug16_3](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EbkEtMzmNU5Em9knfGr2iLABOyNOEjfXwXeBRncGYQABww?download=1)
- [jacobs_Aug16_4_ref](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EdxQp9YoxtNBm3pZeMfiw1gBSYaC9FoXUaukNSEn8dV9Mw?download=1)
- [July16](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EUSLpysLge9EsVAe-r96ToUB_DWHcmMs2-kM_ANeFYWYcg?download=1)
- [July18](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/Eela0I6LUQJNpwj_nBSD4B0BMnbt2ZQnrgqIKFTkznWcaw?download=1)
- [July22_2_ref](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EZySLl-lUIBIiGdpR9tjgksBIEP2jqq4pRshkHxekcPNaA?download=1)

</div>

<div class="col-sm-6 clearfix">

### Feature Downloads:

Cumulative Donloads: [Features Atkinson](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/ET3bKqoYpExCmanla2bLUJEBlC8TXhLL9U2ygNLzMWOrYg?download=1). [Features Jacobs-1](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EbkyAttzBLFMtPlYgkxGMaYB_J6tBExvs1qw8DazkzSdQQ?download=1), [Features Jacobs-2](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/Eeku7wzVdL9FrvhMasCr3D8Ba_YeoZZo7pvU3wkCMglaVA?download=1), [Features Jacobs-3](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/Eb82hHQ5SVdJqqFi4fkC1dYB9o76v43jNT1Df5WHC4tm5A?download=1).

Individual Downloads:

- [jacobs_Jul28](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EYRJAe2dHaRNt6AGTpA9bEkBp7N0lEYScmEzT4HNaNbx1Q?download=1)
- [jacobs_Jul28_2](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EeQf1sXiWehGsD5BCQ08ui8BiDOdhNyq_f7Bf3OHe-lXZw?download=1)
- [jacobs_Aug16_1](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EQ3xv70aECdDguiYTA3px0cBUQCJc5T7WFFrjeb67Ww2CA?download=1)
- [jacobs_Aug16_3](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EUxwVT0kyFBKh2LY45vfrMYBlDApo4Alyr3xzyxOUsf0cQ?download=1)
- [jacobs_Aug16_4_ref](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/ETdyRRQe7UlJg5Aa5VsFf1ABsLj-aQWnRINB2VpHX_6XNw?download=1)
- [July16](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/Eakj2NQkpHRAjOtNUeV6y58B0tgFRVDuRpRnA6os5EXhBw?download=1)
- [July18](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EVI3UwbFH9ZMuk3N0sORXpgBXswXPXJWb5VMl6HW-Tl5ng?download=1)
- [July22_2_ref](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EVkpRrl4ZaxBvyqXN5nWOLYBfPWIYfSvhWN6YeNfKdOXFA?download=1)

</div>

<div class="col-sm-12 clearfix">

### Data Split:

Dataset Split IDs to replicate results from [DLoc](https://wcsng.ucsd.edu/dloc/) using the open-sourced [code](https://github.com/ucsdwcsng/DLoc_pt_code) can be downloaded at [split_ids](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/sayyalas_ucsd_edu/EUHy8MzAmkJAtV5K9CdmpzUBGUs8zU6vl-FQNCmtyNv2Fg?download=1).
	
Folder Metadata:
data_split_idx
	
- data_split_ids_<dataset_name>.mat
	- fov_test_idx: MATLAB indices of the points that are selected in **dataset_&lt;dataset_name&gt;.mat** to generate **dataset_fov_test_&lt;dataset_name&gt;.mat**
	- fov_train_idx: MATLAB indices of the points that are selected in **dataset_&lt;dataset_name&gt;.mat** to generate **dataset_fov_train_&lt;dataset_name&gt;.mat**
	- non_fov_test_idx: MATLAB indices of the points that are selected in **dataset_&lt;dataset_name&gt;.mat** to generate **dataset_non_fov_test_&lt;dataset_name&gt;.mat**
	- non_fov_train_idx: MATLAB indices of the points that are selected in **dataset_&lt;dataset_name&gt;.mat** to generate **dataset_non_fov_train_&lt;dataset_name&gt;.mat**
- data_split_ids_&lt;dataset_name&gt;_space_gen.mat
	- test_idx: MATLAB indices of the points that are selected in **dataset_&lt;dataset_name&gt;.mat** to generate **dataset_test_&lt;dataset_name&gt;.mat**. Usually test_idx = [fov_test_idx;non_fov_test_idx]
	- train_idx: MATLAB indices of the points that are selected in **dataset_&lt;dataset_name&gt;.mat** to generate **dataset_test_&lt;dataset_name&gt;.mat**. Usually train_idx = [fov_train_idx;non_fov_train_idx]

</div>
	
	
	
</div>

#### All the dataset downloads are **PASSWORD** protected. To get the password, please read and agree to the [terms and conditions](https://forms.gle/WWGymUFxhPWRc4zu7). You can then proceed to download datasets from the links above.

---

## CITATION ##

- Ayyalasomayajula R, Arun A, Wu C, Sharma S, Sethi AR, Vasisht D, Bharadia D. Deep learning based wireless localization for indoor navigation. InProceedings of the 26th Annual International Conference on Mobile Computing and Networking 2020 Apr 16 (pp. 1-14).

  [Bibtex]({{ site.url }}{{ site.baseurl }}/files/dloc_bib.md)
  
