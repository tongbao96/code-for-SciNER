

# **A Type-driven Multi-task Learning Framework for Scientific Named Entity Recognition with Large Language Models**

## Overview

**Dataset and source code for paper: "A Type-driven Multi-task Learning Framework for Scientific Named Entity Recognition with Large Language Models".**

## Directory structure
<pre> SciNER                                   Root directory
  ├── data                                Dataset folder
  |   ├── SciERC                          Download SciERC dataset
  |   ├── jnlppa	                  Download JNLPBA dataset
  |   ├── bc5cdr                          Download BC5CDR dataset
  |   ├── generate_bio.py                 Generate traing data for JNLPBA and BC5CDR
  |   ├── generate_sci.py                 Generate traing data SciERC
  |   ├── utils                       
  |   |   ├── tagmap.py                   Entity type mapping file
  ├── SciBERT                             Download SciBERT model
  ├── BioBERT                             Download BioBERT model
  ├── flan-t5-xxl                          Download Flan-t5-xl model
  ├── utils                              
  |   ├── tag_map.py 			  Entity type mapping file
  ├── demonstractions.py                  Code for select demonstractions
  ├── fine_tune_encoder.py                Code for fine_tune_encoder
  ├── history_config.json                 Save model performance
  ├── gpt.py                              Code for test ChatGPT or GPT4
  ├── train.py                            Code for train the model
</pre>

## Dataset Discription

**SciERC** dataset is designed for the computer science domain and contains 6 entity types:Task, Method, Metric, Material, Generic, and Others.  It was introduced in the paper [Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction - ACL Anthology](https://aclanthology.org/D18-1360/))[1]. 

**JNLPBA** focuses on the biomedical domain with 5 entity types: Protein, DNA, RNA, Cell Line, and Cell Type.  It was  introduced in the paper [Introduction to the bio-entity recognition task at JNLPBA](https://dl.acm.org/doi/abs/10.5555/1567594.1567610) [2].

**BC5CDR** targets chemical and disease entity recognition in the biomedical domain.  It was  introduced in the paper [Assessing the state of the art in biomedical relation extraction: overview of the BioCreative V chemical-disease relation (CDR) task ](https://academic.oup.com/database/article/doi/10.1093/database/baw032/2630271) [3]. 

The data from JNLPBA and BC5CDR should follow the format:

```
{ 
  "tokens": [],
  "tags" | "ner_tags": []
}
```

The data from SciERC should follow the format:

```
{
  "tokens": [],
  "entities": [
    {
      "start": start,
      "end": end,
      "type": type
    }
  ]
}
```

## Requirements

We recommend using Anaconda to create your own virtual environment, then install the package below, and ensure you have enough GPU memory.

- python==3.10
- torch==1.11
- numpy==1.26
- transformers==4.43
- peft==0.12
- uniem==0.3
- accelerate==0.33
- datasets==2.20
- bitsandbytes==0.43
- nltk==3.9
- tqdm==4.66
- openai==1.42.0

## Quick start

1. Download the datasets to the `data` directory from their papers. 
2. Download the SciBERT model from (https://huggingface.co/allenai/scibert_scivocab_uncased) to the `SciBERT` directory, the BioBERT model from (https://github.com/naver/biobert-pretrained) to the `BioBERT` directory,and the Flan-t5-xl model from (https://huggingface.co/philschmid/flan-t5-xxl-sharded-fp16) to the `flan-t5-xxl` directory. 
3. Run `generate_bio.py` to generate the training data format for the JNLPBA and BC5CDR datasets, and run `generate_sci.py` to generate the training data format for the SciERC dataset. Adjust the parameters according to your needs.
4. Run `python fine_tune_encoder.py` on different datasets to obtain better vector representations.  You can change the parameters as needed.
5. Run `python lora.py` to train the flan-t5-xl model. This has been tested on a GPU with 40G memory.  You can adjust the parameters as needed.
6. Run `python demonstrations.py` to obtain suitable few-shot examples, then use the above-trained model for inference.
7. For ChatGPT or GPT-4, run `python gpt.py` to call the API for testing. Adjust your own parameters and API key, and pay for the token costs.

## Notes

1. If you are working with datasets from other domains, you can find a suitable encoder at (https://huggingface.co/spaces/mteb/leaderboard). You can choose one based on your needs and hardware settings.
2. For GPT-4, You can  apply for your apikey from (https://openai.com/) and pay for API usage. 
3. Based on the official website, the pricing for GPT-4 is**$30/1M** for input token and **$60/1M** for output tokens. Additional details about the GPT-4 interface or errors encountered can be found at (https://platform.openai.com/docs/api-reference)

## Reference

1. Luan, Y., He, L., Ostendorf, M., & Hajishirzi, H. (2018). Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 3219-3232).
2. Collier, N., Ohta, T., Tsuruoka, Y., Tateisi, Y., & Kim, J. D. (2004). Introduction to the bio-entity recognition task at JNLPBA. In *Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine and its Applications (NLPBA/BioNLP)* (pp. 73-78).
3. Wei, C. H., Peng, Y., Leaman, R., Davis, A. P., Mattingly, C. J., Li, J., ... & Lu, Z. (2016). Assessing the state of the art in biomedical relation extraction: overview of the BioCreative V chemical-disease relation (CDR) task. *Database*, *2016*.

## Citation

Please cite the following paper if you use this code and dataset in your work.

>Tong Bao, Yi Zhao, Heng Zhang, Chengzhi Zhang\*.A Type-driven Multi-task Learning Framework for Scientific Named Entity Recognition with Large Language Models. ***Information processing & management***, 2024(submit).
