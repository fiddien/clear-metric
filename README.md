# Clear Sentences Metric

Check whether a sentence adhere to five principles of clear writing (Williams and Bizup) by leveraging abstract meaning representation parse and dependency parse of the sentence.

Principles:
1. Make the character of the story the subject of the sentence.
2. Make the action of the story the verb of the sentence.
3. Avoid long, abstract subjects.
4. Avoid long introductory phrases and clauses.
5. Avoid interrupting subject and verb connection.

Score 1 indicates violation of the principle, while score 0 indicates adherence to the principle. Thus, lower total score is better.

## Instructions

Clone this repo then move to the folder.
```
git clone https://github.com/fiddien/clear-metric.git
cd clear-metric
```

Create a new virtual environment inside the folder.
```
python -m venv env
```

Install the necessary packages.
```
pip install -r requirements.txt
```

### Prepare the AMR parser model

Create a folder for the AMR parser model.
```
mkdir env/Lib/site-packages/amrlib/data
```

Before continuing, obtain one of the available the AMR parser models in [amrlib](https://github.com/bjascob/amrlib-models). In the following tutorial, the SPRING model is used.

Put the downloaded model file into the `clear-metric` folder. Extract it to the `data` folder.
```
tar -zxvf model_parse_spring-v0_1_0.tar.gz.gz -C env/Lib/site-packages/amrlib/data
```

Rename the folder.
```
mv env/Lib/site-packages/amrlib/data/model_parse_spring-v0_1_0 env/Lib/site-packages/amrlib/data/model_stog
```

### Prepare the syntactic parser model (spaCy)

Download the spaCy model.
```
python -m spacy download en_core_web_sm
```

## Using the script

Run the script to score sentences using the metrics. You can input the sentence through the command itself or from a text file.

To input sentences through the command, use the `-s` flag and separate different sentences using `<sep>`.
```
python main.py -s "Yellow is blue.<sep>Blue befriends green."
```

To read the sentences from a file, use the `-i` flag. Different sentences should be separated by a newline. Output the results into another file by using the `-o` flag.
```
python main.py -i example_sentences.txt -o example_results.txt
```

To only run the action-character detection task, use the `-ac` flag.
```
python main.py -i example_sentences.txt -o example_results.txt -ac
```

To only 

## To Do
- [ ] Issue: The metric currently penalises passive sentences on the first principle. Might want to return multiple action-character pair candidates.
- [ ] Implement batching for both the AMR parsing and syntactic parsing.
- [ ] Implement the "abstract" part of the 3rd principle.
- [ ] Create an XML or JSON format of the output.