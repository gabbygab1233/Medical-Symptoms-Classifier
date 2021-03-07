# Medical Symptoms Text Classification
The adoption of natural language processing in healthcare is rising because of its recognized potential to search, analyze and interpret mammoth amounts of patient datasets. Using advanced medical algorithms, machine learning in healthcare and NLP technology services have the potential to harness relevant insights and concepts from data that was previously considered buried in text form. NLP in healthcare media can accurately give voice to the unstructured data of the healthcare universe, giving incredible insight into understanding quality, improving methods, and better results for patients.

<p align="center">
<img src="https://emerj.com/wp-content/uploads/2018/10/data-mining-medical-records-with-machine-learning-5-current-applications.png"/>
</p>



# [Dataset](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.)
This [Dataset](https://www.kaggle.com/paultimothymooney/medical-speech-transcription-and-intent) contains thousands of audio utterances for common medical symptoms like “knee pain” or “headache,” totaling more than 8 hours in aggregate. Each utterance was created by individual human contributors based on a given symptom. These audio snippets can be used to train conversational agents in the medical field.

This dataset was created via a multi-job workflow. The first involved contributors writing text phrases to describe symptoms given. For example, for “headache,” a contributor might write “I need help with my migraines.” Subsequent jobs captured audio utterances for accepted text strings.

This dataset contains both the audio utterances and corresponding transcription. 

### [Experiment Results:](http://)
* **Data Analysis**
    * Age column contains outliers
    * There's 269 duplicates data
 * **Performance Evaluation**
    * Splitting the dataset by 80 % for training set and 20 % validation set.
 * **Training and Validation**
    * Extra Trees Classifier has a higher accuracy score than the other models achieving 93 %
 * **Fine Tuning**
    * Using  {'criterion': 'entropy', 'max_depth': 15, 'max_features': 'sqrt', 'n_estimators': 10} for Extra Trees Classifier improved the accuracy by 1 %.
 * **Performance Results**
    * Validation Score: 96%
    * ROC_AUC Score: 92 %
   
  # [Demo](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.)
  
  ![](https://i.imgur.com/SJPzebD.png)

# References
* https://www.foreseemed.com/natural-language-processing-in-healthcare
* https://appen.com/datasets/audio-recording-and-transcription-for-medical-scenarios/
* https://www.kaggle.com/paultimothymooney/medical-speech-transcription-and-intent
* https://towardsdatascience.com/medical-text-analytic-techniques-and-its-applications-c7419c7c1f6
* https://github.com/Jcharis/Streamlit_DataScience_Apps/tree/master/news_classifier_nlp-app
