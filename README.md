# Instructions
In this project we are going to bring everything together in a complete ML pipeline that
produces a trained Decision Tree model about Heart Failure Prediction.

## Data Informations
* [Heart Failure Prediction - Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data); 

Heart Failure Prediction: Analysis and modeling. There are some factors that affects Death Event. This dataset contains person's information like age, sex, blood pressure, smoke, diabetes, ejection fraction, creatinine phosphokinase, serum creatinine, serum sodium, time and we have to predict their DEATH EVENT.

## Execution Steps

The starter kit contains all the steps we have previously completed, only slightly modified to 
work better together.

A few notes and instructions:
* When chaining together the steps, the output artifact of a step should be the input artifact
  of the next one (when applicable). Also use the ``artifact_type`` options so that the final
  visualization of the pipeline highlights the different steps. For example, you can use
  ``raw_data`` for the artifact containing the downloaded data, ``preprocessed_data`` for the
  artifact containing the data after the preprocessing, and so on.
  
* For testing, set the ``project_name`` to ``week_09_example_01``. Once you are done
  developing, do a production run by changing the ``project_name`` to 
  ``high_income_classification_prod``. This way the visualization of the pipeline will not contain all your trials and errors. Remember to tag the produced model export as ``prod`` (we are going to use it in the next exercise)
  
* When developing, you can override the parameter ``main.execute_steps`` to only execute one or
  more steps of the pipeline, instead of the entire pipeline. This is useful for debugging. 
  For example, this only executes the ``decision_tree`` step:
  ```bash
  mlflow run . -P hydra_options="main.execute_steps='decision_tree'"
  ```
  and this executes ``download`` and ``preprocess``:
  ```bash
  mlflow run . -P hydra_options="main.execute_steps='download,preprocess'"
  ```
### Used tools
*   [Visual Studio Code](https://code.visualstudio.com/);
*   [Jupyter](https://jupyter.org/);
*   [Google Colab](https://colab.research.google.com/);
*   [Anaconda - 3](https://www.anaconda.com/products/individual);
*   [Wandb](https://wandb.ai);
*   [MLflow](https://mlflow.org/);
*   [Hydra](https://hydra.cc/docs/intro/).

### Authors
*   [Kathleen Noemi](https://github.com/kathleenrego)
*   [Pedro Henrique](https://github.com/pedrohfonseca)
