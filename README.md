# The CL-Eval Framework
### Augmenting Entity-Level Sentiment Analysis with Coreference Resolution
___

## setup

### Models
- Download SpanBERT large [here](https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz)
  - extract the `.tar.gz` file into a created "pretrained/allen-spanbert-large" directory.
- Download CoreNLP
  - Follow instructions [here](https://stanfordnlp.github.io/CoreNLP/download.html)
  - The CoreNLP server is required to run the coreference modules in the `Model Wrappers` directory.
- Download NeuralCoref
  - For the pip version, this can be installed by:
    - `pip install neuralcoref`
    - Further instructions will be prompted

___
### Evaluation
Most of the evaluation happens in the [Evaluation Notebook](CorefLiteEvaluation.ipynb).
These require access to the datasets, which are set to be downloaded in the parent directory of where the `CL-Eval` framework is installed, in a folder named `coreference_data`.
Data may be sent upon request, but some datasets, e.g. OntoNotes and PreCo, are licensed, and can be obtained through https://catalog.ldc.upenn.edu/LDC2013T19 and https://preschool-lab.github.io/PreCo/.
The datasets Litbank and GUM are openly available: https://github.com/dbamman/litbank https://github.com/amir-zeldes/gum


Datasets can obviously not be run without converting them to the `.coreflite` format beforehand, and requires the user to run the notebooks found in the `CorefLite` directory. Each notebook is fairly self-explanatory, and can be updated with the required paths.
