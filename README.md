# <img src="https://github.com/Aditya-R-Chakole/eSeller/blob/main/seller-png.png?raw=true" width="30" height="30"/> [eSeller]

## Question-Answering System for Amazon Products

### Introduction 
**eSeller** is a Streamlit web application, developed to **answer queries regarding any Amazon product**. Given an Amazon product link, this application **scrapes publicly available data from the product page**. Then this scraped data is used to answer questions using the **DistilBERT Transformer model** (from Hugging Face).  

### Table of Content
- **app.py** : Streamlit App 
- **requirements.txt** : Lists all the required libraries
- **runtime.txt** : Lists the python version 
- **startPage1.png / startPage2.png / startPage3.png** : Start-page image

### Requirements
- [Python 3.9.6]
- [Streamlit]
- [BeautifulSoup]
- [Huggingface Transformers]

### RUN
> streamlit run app.py

### Instructions 
- Select a **AMAZON** product
- Put the **Product Link** in the eSeller
- **Verify** the Product and **Ask Any Product related question**.
- Get the **Answer**, or else **try changing the key word**.

### [Live Demo]
![Image](https://github.com/Aditya-R-Chakole/eSeller/blob/main/startPage1.png?raw=true "Select a Ecommerce Website and put the Product Link.")

![Image](https://github.com/Aditya-R-Chakole/eSeller/blob/main/startPage2.png?raw=true "Verify the Product and Ask Any Product related question.")

![Image](https://github.com/Aditya-R-Chakole/eSeller/blob/main/startPage3.png?raw=true "Get the answer, or else try changing the key word.")
### License
[MIT]

[Python 3.9.6]: <https://www.python.org/downloads/release/python-396/>
[Streamlit]: <https://streamlit.io/>
[BeautifulSoup]: <https://pypi.org/project/beautifulsoup4/>
[Huggingface Transformers]: <https://huggingface.co/docs/transformers/index>
[Live Demo]: <https://share.streamlit.io/aditya-r-chakole/eseller/main/app.py>
[eSeller]: <https://share.streamlit.io/aditya-r-chakole/eseller/main/app.py>
[MIT]: <https://github.com/Aditya-R-Chakole/Algorithm_Visualizer/blob/main/LICENSE>
