Hello, I am sharing this information for those who aren't aware about these or looking for the answers.

## For AIML model train :-
There mainly 2 best free jupyter notebook cloud platform for model train with free GPU

### (For concise note scroll down....)

### 1) Google Colab
- Colab offers free NVIDIA Tesla T4 GPU 
- VRAM:- 16GB GDDR6
- CUDA cores :- 2560
- Memory bandwidth:-  320 GB/s
- Tensors core:- 320
- FP32 Performance:- ~8.1 TFLOPS
- FP16 Performance:- ~65 TFLOPS (with Tensor Cores)
- Memory Bus:- 256-bit

### 2) Kaggle
Kaggle offers free NVIDIA Tesla T4x2 GPU and NVIDIA Tesla P100

T4 X 2:- ~32gb VRam but extra code cell to use it.

P100:-
- VRAM:- 16-17GB HBM2
- CUDA Cores:- 3584
- Memory bandwidth:- 732 GB/s
- Tensors core:- 0
- FP32 Performance:- ~9.5 TFLOPS
- FP16 Performance:- ~18.7 TFLOPS 
- Memory Bus:- 4096-bit

### My Thoughts:-
- In simple words P100 is stronger and powerful GPU,its built for datascience,transformer,heavy work reason being it has HBM.

- I personally have shifted to Kaggle for its **high gpu power and consistent connection.**
- Kaggle has total of **30hr** gpu usage limit(weekly) for free and 9 hour a session means uninterrupted training got long hours. Also kaggle training works even **tab is closed**(best for overnight training) unlike Colab.

- Colab has **daily free gpu usage limit** but its depends on how heavy training is going and also not that consistent as Kaggle, its not fixed. **Can't train if tab is closed.** It has google drive mount support.
- Colab is better for inference and testing your model. Easy and fast.

## How to use:- 

## **[Kaggle](https://www.kaggle.com/)** :-
Left section->Code->Notebook,create or  access save notebook(your work).->Settings->Accelerator-> Choose the gpu.

You can view resource on clicking on Draft Session (left of 3 dots)

For t4 x 2 check if both gpu are utilising or not.

## **[Colab](https://colab.research.google.com/)** :-
File->Open or Create new Notebook->Click on downwards arrow below the share option on right->Change Runtime to T4.

You can view resource too under that settings

Which GPU to choose? In some case P100 is just better because of  high memory bandwidth but i suggest asking AI which GPU(T4 x 2 or P100) should you based on project. and **dont forget to ask for code cell to use both t4 gpus**

Both **PyTorch** and **Tensorflow** has compatilibity to run parallel gpu training.

## Dataset,Pre-trained Model
- Kaggle is one of the best source of dataset.
- [Hugging Face](https://huggingface.co/) has datasets and pre-trained models.

## [Roboflow Universe](https://universe.roboflow.com/) :- 
- The best source of dataset and models for computer vision( Image classification, Object Detection, Image Segmentation, OBB, Instance Detection, Key point detectios etc)
- Can make project,and multiple versions of dataset and resize images,image processing, perform data augmentation on the images,there many options of data augmentation. All with just clicks ,no coding needed.
- Can download the dataset or code snippet for for multiple models like YOLO, and make transformers based model like DETR, RF-DETR, and GroundingDINO etc. There bunch of models available.
- Auto label, Manual labels, Professional human labels options are available for your dataset.
- Easy deployment of model and also featured workflow section like n8n.Can use pre-trained model for Inference and testing too.

## Deployment

On Hugging Face you can deploy your project for live demo. 

Also has free TB of storage for you models.

[Render](https://render.com/) is nice for lightweight models,websites.

Same goes for [Vercel](https://vercel.com/), [Netlify](https://www.netlify.com/) and [Railway](https://railway.com/).

Hugging Face is best for **heavy models** and most compatile with [Gradio](https://www.gradio.app/) for easy web framework for your projects.

## Youtube Lecture

- [Campus X](https://www.youtube.com/@campusx-official) 
- [Krish Naik](https://www.youtube.com/@krishnaik06)
- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)
- [3blue1brown](https://www.youtube.com/@3blue1brown) :- Maths and Neural Networks related visualing learning video, easy explaination

- d2l.ai (Dive into Deep Learning) — interactive textbook with code

## Some Github Repo
https://github.com/Developer-Y/cs-video-courses

https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code

https://github.com/DataTalksClub/mlops-zoomcamp


[LMArena](https://arena.ai/) :- Bunch of free AI for use,no signup requires
