This repository is the code for Data Mining HW3. It contains the code for Task 1 and Task 2.
The task 1 code is simply in the root directory and the task 2 code is in the Task2 directory.
The datasets are not uploaded as they are large but below are the locations from which the datasets have been downloaded :
Task 1 : CSE 572 Data Mining course on canvas -> modules -> kmeans_data.zip 
Task 2 : [text](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Note to Grader:

When Task 1 dataset is downloaded, extract it and place the files in the root directory.
For Task 2 dataset, download it and extract it in the Task2 directory.

Install the required packages using the following command :
pip install -r requirements.txt

if you face any error regarding mkl_intel_thread or mkl libraries related errors then please install numpy  

use the command : pip install numpy pandas matplotlib mkl mkl-service

if installing scikit-surprise, use the following command :
conda install -c conda-forge scikit-surprise
Also when running task 2 please make sure to activate the conda environment 

To run the code, use the following commands :
Task 1 : python task1_main.py
Task 2 : python Task2/task2_main.py

Run both the tasks main python file from project root directory.