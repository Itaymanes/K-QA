# K-QA Benchmark
Dataset and evaluation code of K-QA.

This repository provides the dataset and evaluation code for K-QA, a comprehensive Question and Answer dataset in real-world medical . 
You can find detailed information on the dataset curation and evaluation metric computation in our full [paper]().

To explore the results of 7 state-of-the-art models, check out [space](https://huggingface.co/spaces/Itaykhealth/K-QA).

## Loading the dataset

As described in the paper, the dataset comprises 201 questions and answers, incorporating more than 1,589 ground-truth statements. 
Additionally, we provide 1,212 authentic patient questions.

## How to Evaluate New Results
#### Organize Results in a Formatted Way
Before running the evaluation script, ensure that your results are stored in a JSON file with keys `Question` and `result`. Here's an example:
```python
[
  {
  'Question': "Alright so I dont know much about Lexapro would you tell me more about it?",
  'result': "Lexapro is a medication that belongs to a class of drugs\ncalled selective serotonin reuptake inhibitors (SSRIs)"
  }, 
  {
  'Question': "Also what is the oral option to get rid of scabies?" , 
  'result': "The oral option to treat scabies is ivermectin, which is  a prescription medication that is taken by mouth."
  }
]
```

#### Install Requirements
Clone the repository and run the following to install and to activate your virtual environment:
```
poetry install
poetry shell
```
Set keys for GPT-4, either for OpenAI or Azure (the original paper uses models in Azure).
```
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
```
And for Azure also set the following keys:
```
export OPENAI_API_VERSION=""
export OPENAI_TYPE=""
```

Then, run the evaluation script as follows:
```
python run_eval.py 
    --result_file
    --version 
```



#### Cite Us
```markdown
[Include citation information here]
```
