This repo contains the exact free tools I use to:
- Train ML/DL models
- Get datasets
- Deploy projects

No paid stuff,everything free to use.

## Table of Contents
- [Model Training (Free GPUs)](#model-training-)
- [Datasets & Pre-trained Models](#datasets--pre-trained-models)
- [Computer Vision — Roboflow](#computer-vision--roboflow)
- [Deployment](#deployment)
- [YouTube / Learning Resources](#youtube--learning-resources)
- [GitHub Repos](#github-repos)
- [Free AI Tools](#free-ai-tools)
  
## Model Training
There are mainly 2 best free jupyter notebook cloud platform for model train with free GPU

### 1) Kaggle (Best for Training)
- Free GPUs (T4 x2 / P100)
- 30 hours/week usage resets on Saturday 
- Works even when tab is closed 
- Long sessions (~9 hours)

**Best for: Heavy training, long runs**

### 2) Google Colab (Best for quick work)
- Free T4 GPU
- Easy to use
- Google Drive integration
- Not fixed usages hours, depends on load but disconnects often 
- Doesn’t run when tab is closed

**Best for: Testing, small experiments**

### My Thoughts:-
- In simple terms, the P100 is a stronger GPU for data science and transformer workloads.
- The reason is its HBM2 memory much higher bandwidth (732 GB/s vs T4's 320 GB/s) makes a real difference on large models.
- I personally have shifted to Kaggle for it's **high gpu power and consistent connection.**
 
## Quick Setup

**[Kaggle](https://www.kaggle.com/)** Create/open a notebook → Settings → Accelerator → select GPU.
Monitor usage via *Draft Session* panel. For T4×2, use `torch.nn.DataParallel`
or `tf.distribute.MirroredStrategy` to utilize both GPUs, they won't both 
run automatically without this.

**[Colab](https://colab.research.google.com/)** Open notebook → Runtime → Change runtime type → T4 GPU.
Note: training pauses if the tab is closed or session times out.

Which GPU to choose? In some case P100 is just better because of  high memory bandwidth but but I'd suggest asking an AI which GPU(T4 x 2 or P100) should you based on project. and **dont forget to ask for code cell to use both t4 gpus**

Both **PyTorch** and **Tensorflow** has compatibility to run parallel gpu training.

## Datasets & Pre-trained Models
- Kaggle is one of the best source of dataset.
- [Hugging Face](https://huggingface.co/) has datasets and pre-trained models.

## Computer Vision — Roboflow Universe

[Roboflow Universe](https://universe.roboflow.com/) is the best source for computer vision datasets and models (classification, detection, segmentation, OBB, keypoints, and more).

**Dataset Tools**
- Upload, resize, and version your datasets
- Data augmentation with many options, no coding needed
- Auto-label, manual label, or professional human labeling

**Model Export**
- Download datasets or code snippets for YOLO, DETR, RF-DETR, GroundingDINO, and more

**Deployment & Inference**
- Easy model deployment with a featured workflow builder (similar to n8n)
- Use pre-trained models for quick inference and testing

## Deployment

On Hugging Face you can deploy your project for live demo. 

Free unlimited storage for public model and dataset repos.

Hugging Face is best for **heavy models** and most compatible with [Gradio](https://www.gradio.app/) for easy web framework for your projects.

Frontend/web → [Vercel](https://vercel.com/) / [Netlify](https://www.netlify.com/)
Backend/light ML → [Render](https://render.com/) / [Railway](https://railway.com/).

## Youtube Lecture

- [Campus X](https://www.youtube.com/@campusx-official) 
- [Krish Naik](https://www.youtube.com/@krishnaik06)
- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)
- [3blue1brown](https://www.youtube.com/@3blue1brown) :- Maths and Neural Networks. visual explanations, highly recommended for intuition

- d2l.ai (Dive into Deep Learning) — interactive textbook with code

## GitHub Repos
- [CS Video Courses](https://github.com/Developer-Y/cs-video-courses) — curated list of CS courses with video lectures
- [500 AI/ML Projects with Code](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code)
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) — free MLOps course by DataTalks.Club


[LMArena](https://arena.ai/) :- Compare and chat with many AI models for free. Some features may require signup.
