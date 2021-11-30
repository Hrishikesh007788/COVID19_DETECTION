# COVID19 DETECTION USING X-RAY IMAGES

![alt text](https://www.futuretimeline.net/blog/images/1466-chest-xray-ai-technology.jpg)


## Problem Statement:

The quality of concrete is determined by its compressive strength, which is measured
using a conventional crushing test on a concrete cylinder. The strength of the concrete
is also a vital aspect in achieving the requisite longevity. It will take 28 days to test
strength, which is a long period. So, what will we do now? We can save a lot of time and
effort by using Data Science to estimate how much quantity of which raw material we
need for acceptable compressive strength.

## Proposed Solution:

A possible solution would be to create a Machine Learning model which would predict the compressive strength of the concrete given the quantity of the ingredients.  

## Data Description:

Dataset available in kaggle: [Link](https://www.kaggle.com/elikplim/concrete-compressive-strength-data-set)

Sources:
Original Owner and Donor
Prof. I-Cheng Yeh
Department of Information Management
Chung-Hua University,
Hsin Chu, Taiwan 30067, R.O.C.
e-mail:icyeh@chu.edu.tw
TEL:886-3-5186511
Date Donated: August 3, 2007

Data Characteristics:
The actual concrete compressive strength (MPa) for a given mixture under a
specific age (days) was determined from laboratory. Data is in raw form (not scaled).
Summary Statistics:
Number of instances (observations): 1030
Number of Attributes: 9
Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
Missing Attribute Values: None

## Project Tree Structure
```
 .
├── Images
    ├── ineuron-logo.png
├── src
     ├── model_creation.py
     ├── preprocessing.py
├── Concrete Compressive Strength Prediction.ipynb
├── Procfile
├── app.py
├── reports
├── concrete_data.csv
├── requirements.txt
├── templates
├── setup.sh
├── strength.pkl
└── README.md
```

## Tools used:

- Programming language : Python
- IDE : Visual Studio Code
- Visualization : Matplotlib and Seaborn
- Deployment platform : Heroku
- Front end development : HTML/CSS
- Back end development : Streamlit
- Version control system : GitHub

## Web App:

Web App Link: https://compressive-strength-concrete.herokuapp.com

In this web app, we just need to enter the amount of ingredients and the model will give a prediction on the compressive strength of the concrete if made with those amount of ingredients.

## Creator:

1. [Hrishikesh Dutta](https://www.linkedin.com/in/hrishikesh-dutta-6776321a0)



