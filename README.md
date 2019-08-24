<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/thelastpolaris/poors_man_rekognition">
    <img src="resources/static/logo.png" alt="Logo">
  </a>
  <h4 align="center">GSoC 2019 | CCExtractor Development</h4>
</p>

[![Work in Progress](https://img.shields.io/badge/Status-Work%20In%20Progress-Blue.svg)](https://shields.io/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GSoC](https://img.shields.io/badge/GSoC-2019-Red.svg)](https://summerofcode.withgoogle.com/dashboard/project/6506536917008384/overview/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)


  <p align="center">
    <a href="http://fedoskin.org/category/gsoc-2019/"><strong>Track the progress</strong></a> | 
    <a href="http://fedoskin.org/2019/08/24/gsoc-2019-poor-mans-rekognition/"><strong>Final post</strong></a>
  </p>

## About The Project
  This project is done under the umbrella of CCExtractor Development and it was started during Google Summer of Code 2019.
  
  The goal of this project is to create an open-source alternative to Amazon Rekognition, which aims to surpass the capabilities of its proprietary counterpart. The main focus of the current GSoC project is facial detection and recognition. However, in the long term, the aim is to offer a wide array of options ranging from object classification and image segmentation for videos to text analysis of subtitles and conversion of audio to text. 

  Moreover, in order to make it easier to digest and understand vast amounts of the extracted data, this project will offer visualization and analysis tools in the form of a Web application that will allow users to build relationship graphs and get insights from extracted data.
  
## Running PMR

1. Download model files https://drive.google.com/open?id=1qeXTtYQX3-_O--txNGMx0-_Cml5jS8tR Copy it to <i>rekognition</i>, replacing existing <i>model</i> folder
2. Install all the requirements using `pip3 -r requirements.txt`
3. Run `python3 video_pipeline.py -i path/to/your/file`
4. You can find results in <i>output</i> folder

## 
