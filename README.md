# Train, build, and deploy AI/ML projects completely FREE.

This repo contains the exact tools I use to:
- Train ML/DL models
- Find datasets
- Deploy projects

No paid stuff,everything free to use.

## Table of Contents
- [Model Training (Free GPUs)](#model-training)
- [Datasets & Pre-trained Models](#datasets--pre-trained-models)
- [Computer Vision](#computer-vision)
- [Deployment & Optimization](#deployment-&-optimization)
- [YouTube / Learning Resources](#youtube-lecture)
- [GitHub Repos](#github-repos)
- [Free AI Tools](#free-ai-tools)
  
## Model Training
There are 2 commonly used free cloud notebook platforms that provide GPUs:-

### 1) Kaggle (Best for Training)
- Free GPUs **{T4 x2 (30 GB VRAM) / P100 (16 GB VRAM)}**
- 30 hours/week usage resets on Saturday 
- Long sessions **(~12 hours)**
- Use "Save and Run All (Commit)". This let notebook run in background session on Kaggle's servers. You can turn off your laptop, lose internet, or even travel; the notebook will run for up to 12 hours and save your weights automatically.
- System Specs:- **4 CPU cores, 29-30 GB RAM, 50-60 GB Disk Storage**
  
**Best for:- Heavy training, long runs**

### 2) Google Colab (Best for quick work)
- Free T4 GPU **(15 GB VRAM)**
- Easy to use
- Google Drive integration
- Usage hours are not fixed and depend on system load; sessions may disconnect frequently
- Doesn’t run when tab is closed
- System Specs:- **2 CPU cores, 12-13 GB RAM, 100+ GB Disk Storage** 

**Best for:- Testing, small experiments**

### Common Mistake

Mistake: "I closed my tab and my training stopped!"

The Fix: You likely used an Interactive Session. Always use the 'Submit' or 'Save Version' option -> 'Save and Run All' button for actual training.

### My Thoughts
- In simple terms, the T4 x 2 is 2x~ faster than P100 and let you use larger batch.
- It's just P100 give 1-3% better metrics because it's a single gpu and does have to communicate with another gpu.. but it's takes double time than T4 x 2.
- I personally have shifted to Kaggle for its **high GPU power and consistent connection.**

### Which GPU should you choose?

- **T4 x 2**:-Best for anything, it's 2x~ faster and can process larger batch too. (Don't forget to utilise dual gpu)
- **P100**:- Strong gpu if model isn't too heavy, most time just ignore.
- If you're unsure:-start with **T4 x 2**

## Quick Setup

**[Kaggle](https://www.kaggle.com/)** :-
Create/open a notebook -> Settings -> Accelerator -> select GPU.
Monitor usage via *Draft Session* panel. 

For T4×2, use `torch.nn.DataParallel` (PyTorch)
or `tf.distribute.MirroredStrategy` (TensorFlow) to utilize both GPUs, they won't both 
run automatically without this.

In Ultralytic's YOLO, in model training code, set device =[0,1] for dual and [0] for single gpu.

<img width="319" height="180" alt="image" src="https://github.com/user-attachments/assets/f09713f7-bbd0-47a5-8f53-16e3cee9d46e" />

<br><br>

<img width="201" height="546" alt="image" src="https://github.com/user-attachments/assets/1649514e-a94c-4be4-b746-e60bafd6a612" />

<br><br>

**[Colab](https://colab.research.google.com/)** :-
Open notebook -> Runtime -> Change runtime type ->T4 GPU.
<br><br>

<img width="349" height="373" alt="image" src="https://github.com/user-attachments/assets/8922d183-16d2-4689-9d7c-15e0a6a29276" />

<br><br>

<img width="549" height="546" alt="image" src="https://github.com/user-attachments/assets/73f37708-cce6-4ae2-94d8-df5635ed433a" />

<br><br>

<img width="446" height="411" alt="image" src="https://github.com/user-attachments/assets/b5227ed3-eb6a-4646-af59-c7b627445bd9" />

<br><br>

Note: training pauses if the tab is closed or session times out.

## Datasets & Pre-trained Models
- Kaggle is one of the best sources for datasets.
- [Hugging Face](https://huggingface.co/) has datasets and pre-trained models.

## Computer Vision

[Roboflow Universe](https://universe.roboflow.com/) is a great platform for computer vision datasets and models (classification, detection, segmentation, OBB, keypoints, and more).

**Dataset Tools**
- Upload, resize, and version your datasets
- Data augmentation with many options, no coding needed
- Auto-label, manual label, or professional human labeling

**Model Export**
- Download datasets or code snippets for YOLO, DETR, RF-DETR, GroundingDINO, and more

**Deployment & Inference**
- Easy model deployment with a featured workflow builder (similar to n8n)
- Use pre-trained models for quick inference and testing

## Deployment & Optimization

### Model Optimization (The "Secret" to Smooth Demos)
- Don't deploy raw training model file (.pt, .keras, .h5) directly. They are heavy and slow. I suggest always optimize your model first.
- Export to ONNX: Use the ONNX (Open Neural Network Exchange, an universal format) format to reduce latency, with precision f16 it gets much faster inference on gpu and for cpu int8 is better with 1-3% less accuracy.
- Export to OpenVINO: Use the OpenVINO for **only intel cpu** for local machine inference with int8 precision for much faster inference with cost of 1-3% accuracy.
- Remove Dependencies: ONNX models run using onnxruntime, which is much lighter (200 MBs) than installing the full PyTorch or TensorFlow library (2-4 GBs) on your server.

### Where to Deploy
- [Hugging Face](https://huggingface.co/spaces)(Best for ML Demos):
- Live Demos: I suggest [Gradio](https://www.gradio.app/) or [Streamlit ](https://streamlit.io/)to build a web interface in pure Python easily.
- Hardware: Provides free CPU basic tiers. If your model is optimized (ONNX), it will run smoothly even without a free GPU.
- Storage: Free unlimited storage for public model weights and dataset repositories.

[Render](https://render.com/) / [Railway](https://railway.com/).(Best for Backends):
- Great for hosting a FastAPI or Flask web server that serves your model as an API.

[Vercel](https://vercel.com/) / [Netlify](https://www.netlify.com/) (Best for Frontends):
- Use these if you built a custom React/Vue/Next.js frontend to talk to your Hugging Face or Render backend.

## YouTube / Learning Resources

- [Campus X](https://www.youtube.com/@campusx-official) :- In depth Hindi lectures covering Math for ML, classical ML, deep learning, LLMs, and Agentic AI. Includes practical project implementations. Best structured course channel in Hindi.

- [Krish Naik](https://www.youtube.com/@krishnaik06) :- Similar to Campus X but in Hinglish. Good for end-to-end project walkthroughs and staying updated on new tools/frameworks.

- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) :- Ex-OpenAI/Tesla. Builds neural networks from scratch (GPT, tokenizers, backprop). Best channel if you want deep intuition over how things actually work.

 - [3Blue1Brown](https://www.youtube.com/@3blue1brown) :- Visual explanations of Maths and 
  Neural Networks. Highly recommended for building intuition before diving into code.

- [d2l.ai](https://d2l.ai) :- Dive into Deep Learning. Interactive textbook with theory with runnable code. Covers everything from linear models to transformers.

## GitHub Repos
- [CS Video Courses](https://github.com/Developer-Y/cs-video-courses) :— Detailed list of CS courses with video lectures
- [500 AI/ML Projects with Code](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code)
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) :— Free MLOps course by DataTalks.Club

## Free AI Tools

[LMArena](https://arena.ai/) :- Compare and chat with many AI models for free. Some features may require signup.

  
## My Workflow

1. Dataset :- Kaggle / Hugging Face  
2. Data Processing :- Roboflow (for computer vision tasks)  
3. Training :- Kaggle (T4 x 2)
4. Testing :- On local cpu/t4 gpu (ONNX Fp16/int8 format)
5. Deployment :- Hugging Face Spaces / Render  

This is the exact pipeline I use for most of my projects.

## Why this repo?

I created this to help people who:
- Don’t have high-end laptops
- Can’t afford paid GPUs
- Still want to build real AI/ML projects

Everything here is tested and actually used by me.

If this repo helps you, consider giving it a star. ⭐ 
