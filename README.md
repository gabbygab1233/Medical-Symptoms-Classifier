# Early-Stage-Diabetes-Risk-Predictor
Diabetes is a chronic, metabolic disease characterized by elevated levels of blood glucose (or blood sugar), which leads over time to serious damage to the heart, blood vessels, eyes, kidneys and nerves. The most common is type 2 diabetes, usually in adults, which occurs when the body becomes resistant to insulin or doesn't make enough insulin. In the past three decades the prevalence of type 2 diabetes has risen dramatically in countries of all income levels. Type 1 diabetes, once known as juvenile diabetes or insulin-dependent diabetes, is a chronic condition in which the pancreas produces little or no insulin by itself. For people living with diabetes, access to affordable treatment, including insulin, is critical to their survival. There is a globally agreed target to halt the rise in diabetes and obesity by 2025.

<p align="center">
<img src="https://northmemorial.com/wp-content/uploads/2016/10/Diabetes-illustration.png"/>
</p>

About 422 million people worldwide have diabetes, the majority living in low-and middle-income countries, and 1.6 million deaths are directly attributed to diabetes each year. Both the number of cases and the prevalence of diabetes have been steadily increasing over the past few decades.

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
    

# References
* https://www.foreseemed.com/natural-language-processing-in-healthcare
* https://appen.com/datasets/audio-recording-and-transcription-for-medical-scenarios/
* https://www.kaggle.com/paultimothymooney/medical-speech-transcription-and-intent
