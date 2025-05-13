# YOLO11-Earth: Training a Toy Remote Sensing Open Vocabulary Detector at only $21 :exclamation: :coffee:
This an English technical report for my final project in my undergraduate days 🎓.  
Also, this project might be my last computer vision one. So I decided to build this repo and share my interesting findings. :star2:
## the Technical Report pdf
The technical report is in [`tech report`](https://github.com/CatManJr/Train-an-OVD-within-21-dollars/tree/main/tech%20report), and the latex project is provided.
Why ***$21*** ? That means the biggest model YOLO11x-Earth is pretrained on xView on a RTX 4090 for less than 70 hours.
The price of renting a RTX 4090 is $0.3/h on AudoDL.  
At first, I want to publish it on Arxiv. But it's a pity that I do not have the endorsement required.
It's framework:

<img src="https://github.com/CatManJr/Train-an-OVD-within-21-dollars/blob/main/tech%20report/image/5.png" width="50%" style="display: block">  

It's performance after finetuning on DIOR:

<img src="https://github.com/CatManJr/Train-an-OVD-within-21-dollars/blob/main/tech%20report/image/6.png" width="30%" />   

## Code
I provide the model config file `yolo11-earth.yaml` in [`model`](https://github.com/CatManJr/Train-an-OVD-within-21-dollars/tree/main/model) and some useful toolkits in `utils` , like how to split xView and how to build a gradio app for your model.
There is no furthur plan to share my training scripts. Becasue with the model yaml you can easily reproduce my work with 
![Ultralytics](https://badgen.net/badge/Ultralytics/Open%20Source/blue?icon=github)
## Wieghts
I'm glad to share my weights.<br>
Now I'm looking for a efficient methods to share these weights (about 50 files), maybe Google Drive. 
Let me take a break at first plz. :notes:
