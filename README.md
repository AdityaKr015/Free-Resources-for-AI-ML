# Train, build, and deploy AI/ML projects completely FREE.

This repo contains the exact tools I use to:
- Train ML/DL models
- Find datasets
- Deploy projects

No paid stuff,everything free to use.

---

## Table of Contents
- [Model Training (Free GPUs)](#model-training)
- [Datasets & Pre-trained Models](#datasets--pre-trained-models)
- [Computer Vision](#computer-vision)
- [Deployment & Optimization](#deployment-&-optimization)
- [YouTube / Learning Resources](#youtube-lecture)
- [GitHub Repos](#github-repos)
- [Free AI Tools](#free-ai-tools)

---

## Model Training
Most people know Google Colab and Kaggle, but there's hidden gem **Marimo Molab**:-

| Platform | GPU (VRAM) | CPU/RAM | Session | Background Training | Runtime Disk |Persistent Storage |
|----------|------------|-----------|---------|----------------|----------|------------------------|
| Marimo Molab | RTX Pro 6000 Blackwell (96 GB) | 4/32 GB | Up to 12 hr | **No** | Not officially stated | Yes but quota not public |
| Kaggle | 2× T4 (30 GB) / P100 (16 GB) | 4/30 GB | Up to 12 hr | **Yes** (Save & Run / Commit ) | 50–60 GB | Your own Kaggle Datasets |
| Google Colab | T4 (15 GB) | 2/12 GB | Variable | **No** | ~100 GB  | Google Drive |

> Extra Info

### 1) Marimo Molab (Powerful GPU)

Marimo is an open-source **reactive Python notebook** and Molab is its **free cloud notebook platform** with GPU.

- Built-in connectors for Google Drive, Amazon S3, Google Cloud Storage, Azure Blob Storage, and CoreWeave Object Storage.
- Marimo is a reactive Python notebook:- Run a cell or interact with a UI element, and marimo automatically runs dependent cells (or marks them as stale) keeping code and outputs consistent.
- Marimo notebooks are stored as pure Python (with SQL support), executable as scripts, and deployable as apps.

**Best for:- Heavy LLM Finetuning, computer vision, long run with high VRAM**

### 2) Kaggle (A Balanced Platform)
- 30 hours/week usage resets on Saturday 
- Kaggle Datasets can be attached directly to notebooks without downloading, making dataset loading extremely fast.
- Or you can easily upload your custom dataset/ model and use in notebook.
- Supports background execution through **Save Version -> Save & Run All (Commit)**.
- Excellent community, competitions, notebooks and discussions.
  
**Best for:- Medium parameters modeltraining, long runs**

### 3) Google Colab (Best for quick work)
- Usage hours are not fixed and depend on system load, sessions may disconnect frequently
- Doesn’t run when tab is closed
- Native Google Drive integration.
  
**Best for:- Testing, small experiments**

### Kaggle Tip

Mistake:- "I closed my tab and my training stopped!"

The Fix:- You likely used an Interactive Session. Always use the 'Submit' or 'Save Version' option -> 'Save and Run All' button for actual training.

### My Thoughts

- If Molab is available, I'd choose it first because of the massive 96 GB VRAM.
- For most training jobs, I still prefer Kaggle because of its stable sessions, background execution and excellent ecosystem.
- I mainly use Colab for quick experiments and inference.

### Which GPU should you choose?

- **RTX Pro 6000 (Blackwell):** Currently the most powerful free cloud GPU. Ideal for LLM fine-tuning, large vision models, diffusion models, and workloads requiring very high VRAM (96 GB).
- **2× T4:** Strong for most computer vision and deep learning projects. Just make sure the code can actually utilize both GPUs. (DataParallell)[PyTorch]/(MirroredStrategy)[Tensorflow]
- **P100:-** Capable GPU if your model fits within 16 GB VRAM, though the dual T4 option is generally the better choice.
- If you're unsure:- Start with **RTX Pro 6000**

---

## Quick Setup

**[Marimo Molab](https://molab.marimo.io/):-**

**How to add Remote storages:-**

<table border="0">
  <tr>
    <td align="center">
      <b>Side Panel->Views Files->Remote Storagr</b><br><br>
      <img src="https://github.com/user-attachments/assets/5009851b-0e0b-4649-b8d3-d23d3dbfa550">
      <br>
    </td>
    <td align="center">
      <b>Add Remote Storage</b><br><br>
      <img src="https://github.com/user-attachments/assets/ce51d328-baff-4725-b23d-176a2527acd6">
      <br>
    </td>
  </tr>
</table>

**How to Setup GPU:-**

<table border="0">
  <tr>
    <td align="center">
      <b>Top Middle Panel</b><br><br>
      <img src="https://github.com/user-attachments/assets/764c7e46-b8e4-4a50-a4a7-cd478f4dd6c5">
      <br>
    </td>
    <td align="center">
      <b>Select GPU and Save</b><br><br>
      <img src="https://github.com/user-attachments/assets/197a8595-8b9e-4583-bfd0-052eed7df67f">
      <br>
    </td>
    <td align="center">
      <b>Utilisation</b><br><br>
      <img src="https://github.com/user-attachments/assets/bf3a74ed-b9c5-4d4e-939a-9ea3d4abd26b">
      <br>
    </td>
  </tr>
</table>

**[Kaggle](https://www.kaggle.com/):-**

**How to easily load Dataset/Model:-**

<table border="0">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/be02b54d-9495-47bc-83cf-02522c5c5992" width="400">
      <br>
      <sub>From here you can load dataset/models(Add Input).Also can "Upload" Dataset/Model</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/e3b1da55-b363-462f-b265-e2db83b75e4f" width="400">
      <br>
      <sub>Use "Add Input" for filters</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/3074471e-dfd6-44a1-b429-18aae09d7f0f" width="400">
      <br>
      <sub>Filters to access your uploaded Dataset/Model.</sub>
    </td>
  </tr>
</table>


**How to setup GPU:-**

Create/open a **notebook -> Settings -> Accelerator -> select GPU**.
Monitor usage via *Draft Session* panel. 

For T4×2, use `torch.nn.DataParallel` (PyTorch)
or `tf.distribute.MirroredStrategy` (TensorFlow) to utilize both GPUs, they won't both 
run automatically without this.

In Ultralytic's YOLO, in model training code, set `device =[0,1]` for dual and `[0]` for single gpu.

<table border="0">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f09713f7-bbd0-47a5-8f53-16e3cee9d46e" width="400">
      <br>
      <sub>Select GPU Accelerator in Settings</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/1649514e-a94c-4be4-b746-e60bafd6a612" width="200">
      <br>
      <sub>Monitor your VRAM and usage here</sub>
    </td>
  </tr>
</table>

**[Colab](https://colab.research.google.com/)** :-

**How to load Dataset/Model:-**

<table border="0">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/635bcb59-2ff9-4d3d-8991-d0eef2f60faf" width="400">
      <br>
      <sub>Folder icon to access the files of Notebook"</sub>
    </td>
  </tr>
</table>

- Use **'Upload'** icon to Upload Dataset/Model in the temporary files of notebook server.
- **OR** you can save the Dataset/Model on your Google Drive and mount the drive using Drive icon.
- if you prefer running code for mounting

```python
from google.colab import drive
drive.mount('/content/drive')
```

**How to setup GPU:-**

**Open notebook -> Runtime -> Change runtime type ->T4 GPU.**

<br><br>

<table border="0">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8922d183-16d2-4689-9d7c-15e0a6a29276" width="300">
      <br>
      <sub>Click on "Change runtime type"</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/73f37708-cce6-4ae2-94d8-df5635ed433a" width="300">
      <br>
      <sub>Select T4 GPU from the list</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b5227ed3-eb6a-4646-af59-c7b627445bd9" width="300">
      <br>
      <sub>Check resource usage and runtime time remaining by clicking on graph in the corner</sub>
    </td>
  </tr>
</table>

Note:- Training pauses if the tab is closed or session times out, and it also leads to loss of data notebook generated.
Save the data(in zip) locally before closing of notebook.

You can this code in cell to zip the data.

```zip -r YOURZIPNAME.zip DIRECTLYOFFOLDERSTOZIP```

---

## Datasets & Pre-trained Models

### [Kaggle](https://www.kaggle.com/datasets)
Kaggle is one of the largest free dataset repositories on the internet with millions of public datasets.

**What's available:**
- **Tabular / CSV data:-** Great for classical ML (regression, classification)
- **Image datasets:-** For CV tasks like image classification, detection, segmentation
- **Audio and video datasets:-** For speech/media projects
- **NLP datasets:-** Text classification, sentiment, translation
- **Time-series datasets:-** For forecasting tasks

**Why it's useful:**
- Datasets are community uploaded and regularly updated.
- Most popular datasets come with community notebooks, you can see exactly how others loaded and used the data (It's one of the underrated way to learn, to see how experts/experienced people write code).
- Direct integration with Kaggle notebooks means zero download time when training on the platform.

### [Hugging Face](https://huggingface.co/)
Hugging Face is the largest open-source AI platform, think of it as GitHub for ML.

**Datasets**
Covers nearly every ML task:
- **NLP:-** Classification, NER, translation, summarization, Q&A
- **Computer Vision:-** Classification, detection, segmentation, depth, GAN, Stable Diffusion
- **Audio:-** ASR, speaker identification, audio classification
- **Multimodal:-** image-text, video-text, document understanding

Load any dataset in one line:-
```python
from datasets import load_dataset
dataset = load_dataset("dataset-name")
```

**Pre-trained Models**
This is where Hugging Face really shines. Instead of training from scratch (which costs time and compute), you can easily load a pre-trained model and fine-tune it on your specific data.

Some examples:-
- LLMs        :- LLaMA, Mistral, Qwen, Gemma, Phi
- Vision      :- ViT, CLIP, SAM, DETR, RT-DETR
- Audio       :- Whisper, Wav2Vec 2.0, HuBERT
- Multimodal  :- BLIP, LLaVA, Florence, Idefics
- Embeddings  :- sentence-transformers, BGE, E5

**My suggestion:** Before training anything from scratch, always search Hugging Face first. Chances are a model already exists that's 80-90% of the way to what you need. You just need to fine-tune it on your data.

---

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

---

## Deployment & Optimization

### Model Optimization (The "Secret" to Smooth Demos)

Don't deploy your raw training model file (.pt, .keras, .h5) directly. They are heavy and slow. Always optimize your model before deployment.

There are two things I suggest to follow:

**1. Export Format (Changing the architecture)**
   
Exporting converts your model from its training format into a format optimized for inference.

- **ONNX (Open Neural Network Exchange):-** A universal format that works across frameworks and hardware. Models run using `onnxruntime`, which is much lighter (~200 MB) than installing full PyTorch or TensorFlow (~2–4 GB) on your server.

- **OpenVINO:-** Intel's inference format. Only for **Intel CPUs**, but gives significantly faster inference on local machines compared to ONNX.

- **TensorRT:-** NVIDIA's inference format. Only for **NVIDIA GPUs**, best for maximum speed on GPU deployment.

**2. Quantization (Reducing the precision)**
   
Quantization reduces the numerical precision of your model weights, making the model smaller and faster with a small accuracy tradeoff.

|Precision |	Best For	| Speed	| Accuracy Drop|
|---------|------------|----------|--------------|
| **FP32**	| Training (default) |	Slowest |	None |
| **FP16**	| GPU inference	| Fast | ~0-1% |
| **INT8**	| CPU inference	| Fastest |	~1–3% |

*Rule of thumb:-* Use **FP16** when deploying on GPU, **INT8** when deploying on CPU.

**Recommended Combinations:-** 
| Hardware |	Format |	Precision |
|----------|----------|-------------|
| Any CPU |	ONNX |	INT8 |
| Intel CPU |	OpenVINO |	INT8 |
| Any GPU  |	ONNX |	FP16 |
| NVIDIA GPU	| TensorRT |	FP16/INT8 |

*Start with **ONNX**. Only move to **TensorRT** or **OpenVINO** if you need maximum speed.*

### Where to Deploy
- [Hugging Face](https://huggingface.co/spaces)(Best for ML Demos):
- Live Demos: I suggest [Gradio](https://www.gradio.app/) or [Streamlit ](https://streamlit.io/)to build a web interface in pure Python easily.
- Hardware: Provides free CPU basic tiers. If your model is optimized (ONNX), it will run smoothly even without a free GPU.
- Storage: Free unlimited storage for public model weights and dataset repositories.

[Render](https://render.com/) / [Railway](https://railway.com/).(Best for Backends):
- Great for hosting a FastAPI or Flask web server that serves your model as an API.

[Vercel](https://vercel.com/) / [Netlify](https://www.netlify.com/) (Best for Frontends):
- Use these if you built a custom React/Vue/Next.js frontend to talk to your Hugging Face or Render backend.

---

## YouTube / Learning Resources

- [Campus X](https://www.youtube.com/@campusx-official) :- In depth Hindi lectures covering Math for ML, classical ML, deep learning, LLMs, and Agentic AI. Includes practical project implementations. Best structured course channel in Hindi.

- [Krish Naik](https://www.youtube.com/@krishnaik06) :- Similar to Campus X but in Hinglish. Good for end-to-end project walkthroughs and staying updated on new tools/frameworks.

- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) :- Ex-OpenAI/Tesla. Builds neural networks from scratch (GPT, tokenizers, backprop). Best channel if you want deep intuition over how things actually work.

 - [3Blue1Brown](https://www.youtube.com/@3blue1brown) :- Visual explanations of Maths and 
  Neural Networks. Highly recommended for building intuition before diving into code.

- [d2l.ai](https://d2l.ai) :- Dive into Deep Learning. Interactive textbook with theory with runnable code. Covers everything from linear models to transformers.

---

## GitHub Repos
- [CS Video Courses](https://github.com/Developer-Y/cs-video-courses) :— Detailed list of CS courses with video lectures
- [500 AI/ML Projects with Code](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code)
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) :— Free MLOps course by DataTalks.Club

---

## Free AI Tools

[LMArena](https://arena.ai/) :- Compare, test, and chat with dozens of open-source and proprietary LLMs for free side-by-side.

[Gamma AI](https://gamma.app/) :- Generate beautiful presentation slides using simple text prompts. I use it for presenting  ML projects.

*I suggest using [temp email.](https://temp-mail.org/) to keep your main inbox free from promotional spam.*

---

## My Workflow

1. **Dataset :-** Kaggle / Roboflow/ Hugging Face  
2. **Data Processing :-** Roboflow (for computer vision tasks)  
3. **Training :-** Marimo Molab/Kaggle
4. **Testing :-** On local CPU/T4 gpu (ONNX FP16/INT8 format)
5. **Deployment :-** Hugging Face Spaces 

This is the pipeline I use for most of my projects.

## Why this repo?

I created this to help people who:-
- Don’t have high-end laptops
- Can’t afford paid GPUs
- Still want to build real AI/ML projects

Everything here is tested and actually used by me.

If this repo helps you, consider giving it a star. ⭐ 
