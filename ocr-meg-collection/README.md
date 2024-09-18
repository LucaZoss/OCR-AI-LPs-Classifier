![Background](background_img.jpeg)
# Catalog Labelling AI App - *using OCR & Generative AI*
**INDEX**

- [Project Overview](#Project-Overview)
- [How-To-Guide](#How-to-Guide)
- [Further Improvements](#Further-Improvements)
- [TO-DO](#to-do)

## Project Overview

Hi there! This is a project overview of a tool created to help me do Vinyl LPs catalog labelling within my switzerland citizen labourship (kind of swiss military service.)

🎯 **Project Objective** 

The goal of this application was to optimize the time used for labelling the collection inside the MEG museum (ethno-musicologie department).

The initial situation was to do this work manually, making on average from 8 to 10 LPs per day (~1.5hour per LP). This project aims to reduce this work to 20minutes per object. (Making it 5.5x faster!⚡️) (This calculation is taking into account human feedbacks in the loop.)

💿 **Version 1**

The first version of this application should be approached as a companion for labelling the LPs, therefore a manual process of double checking and editing should be done manually. In future version, we could imagine to switch to a more agentic-based methodology plugin some API such as DISCOGS or event Google Search?

The core of the application hide in the ```ocr_meg_collection```folder, in which its architecture is settled as follows:


**Application Archtecture**

![Architecture_Image](architecture.png)

1.ocr_meg_collecton --> Poetry Orchestration Environement
```bash
. // Poetry Ochestration Environement == ocr_meg_collection
├── __init__.py
├── ai_classification_inf.py
├── ai_classification_inf_debug.py
├── cleaner.py
├── main.py
├── ocr_pipeline.py
├── post_processing.py
├── post_processing_debug.py
└── utils.py
```
**Cost**

In terms of cost, I still have the 200$ of new account bonus of GCP credits which is quite cool for experimenting!

In theory this is the breakdown of the cost:

| Service       | Cost (USD) | Description |
|------------------|------------------|------------------|
| OCR (Google Cloud Vision API)| $1.5/1000 Units           |First 1000 units/month are Free! From 1001-5M|
| AI Inference (Gemini 1.5 Pro)       | $0.005-$0.008           |text input *(small context window)* = $0.00125 / 1k char. ; text output = $$0.00375 / 1k char.
|**Totals**     | **For 1 LP**      | **For 1000 LPs**           |
| with free OCR    | **$0.0065**      | **$6.5**           |
| after free-tier    | **$0.008**      | **$8**           |




**Limits & Food for thoughts**

Through the pre-prod phase, I have tried to use different methods for the OCR and the AI inference, using more opensource libraries:

OCR : Pytesseract ; EasyOCR

AI Inference: Ollama Instruct Model (Mistral Nemo, llama3.1)

Also I have tested a full vision LLM with GPT 4o. However the results wasn't as goog in terms of response quality than the paid model used.

One other limits are the fact that running the script locally as we are making API inferences, we have some latency that is manageable (we could as well increase the quotas that are now set to 10 for the AI inference.). Some option would be either to pass to local softwares for OCR + AI_Inference with Ollama or Groq in order to use some GPUs. Or another but more costly option could be to containerize the app and run it through a Cloud Run Instance in GCP (Info to be checked?)

## Install & How-to Guide

-- clone repo
-- add poetry or install requirements.txt






## TO-DO

- [x] Publish first version
- [ ] Create README File
- [ ] Dockerize
- [ ] Fine-Tune Output
- [ ] Ollama?
#
![Logo](MEG.jpg)




[def]: ocr-meg-collection/architecture.png