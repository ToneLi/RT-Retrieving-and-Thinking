## How far is Language Model from 100% Few-shot Named Entity Recognition in Medical Domain
This is the source code of the model RT (Retrieving and Thinking). For the full project, please check the file RT_BC5CDR/3_RT and RT_NCBI/3_RT, the implementation of GPT-NER  and PromptNER is in the BC5CDR.zip and NCBI.zip.
we refer to the source of [code of GPT-NER](https://github.com/ShuheWang1998/GPT-NER) and [paper of GPT-NER ](https://arxiv.org/abs/2304.10428) in our project and the implementation of GPT-NER.

### 1) Overview

The architecture of our proposed RT is depicted in the diagram below.   It consists of two major parts:

<img src="https://github.com/ToneLi/RT-Retrieving-and-Thinking/blob/main/RT_framework.png" width="500"/>
Based on the findings mentioned above, we introduce a novel approach called RT (Retrieving and Thinking) and present it in Figure 5. The RT
method comprises two primary steps: (1) retrieving the most pertinent examples for the given test sentence, which are incorporated as part of the instruction 
in the ICL. This step is accomplished through the process of Retrieving. (2) guiding LLM to recognize the entity gradually, demonstrating this
progression as Thinking. In the following sections, we provide a comprehensive explanation of each component.


### 2) Run Code

```markdown
 For example: 

 model generation:
 please enter: RT_BC5CDR\3_RT

 step1:  python 0_extract_mrc_knn.py
 step2: python 1_extract_mrc_knn5.py
 step3: 2_our_model_shot1_BC5.py
 step4:2_our_model_shot5_BC5.py 
 Please use the default parameters.

model test:
 please enter RT_BC5CDR\3_RT\data\BCD5\
 python F1_evaluation.py
```

### 3) Examples in Different Instructions 
In Figure. 2, we give a detailed example of output on different ICL strategies, Vanilla ICL, the chain
of thought, and the tree of thought. For the idea of the tree of thought, we design the
examples (samples) in the instruction as shown in below. 

<img src="https://github.com/ToneLi/RT-Retrieving-and-Thinking/blob/main/different_output.png" width="500"/>

