## Importing required Libraries
import streamlit as st

import numpy as np
import pandas as pd
import requests
import json
import re
from bs4 import BeautifulSoup as bs

from tensorflow.python.keras import models, layers, optimizers
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizer
from textblob import TextBlob
import torch
import textwrap

## storing models into cache
@st.cache(ttl = 3600)
def load_model( ):
    return DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

@st.cache(ttl = 3600)
def load_tokenizer( ):
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)

## Function to answer the 'question' based on the given 'context'
def qna_bert(context, question):
    model = load_model()
    tokenizer = load_tokenizer()
        
    def check_spelling(question):
        question = re.sub(r'[^\w\s]', '', question)
        question = question.lower()
        question_list = question.split()

        for i in range(len(question_list)):
            question_list[i] = str( TextBlob(question_list[i]).correct() )
        
        question = " ".join(question_list)
        return (question + " ?")

    def answer_question(question, answer_text):
        encoding = tokenizer.encode_plus(question, answer_text)
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outputs = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

        return answer_tokens_to_string

    context = context
    question = check_spelling(question)
    answer = answer_question(question, context)

    return {'context': context, 'question' : question, 'answer' : answer}

## Function to Scrape product related data 
def scrape_data(productURL):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    productPage = requests.get(productURL, headers=headers)
    productSoup = bs(productPage.content,'html.parser')
    
    # Product-Name
    productNames = productSoup.find_all('span', id='productTitle')
    if len(productNames) > 0:
        productNames = productNames[0].get_text().strip()
    
    # Offer-Price
    ids = ['priceblock_dealprice', 'priceblock_ourprice', 'tp_price_block_total_price_ww']
    for ID in ids:
        productDiscountPrice = productSoup.find_all('span', id=ID)
        if len(productDiscountPrice) > 0 :
            productDiscountPrice = productDiscountPrice[0].get_text().strip().split('.')[0]
            productDiscountPrice = productDiscountPrice +'.00'
            break
    
    # MRP-Price
    classes = ['priceBlockStrikePriceString', 'a-text-price']
    for CLASS in classes:
        productActualPrice = productSoup.find_all('span', class_=CLASS)
        if len(productActualPrice) > 0 :
            productActualPrice = productActualPrice[0].get_text().strip().split('.')[0]
            productActualPrice = productActualPrice + '.00'
            break
    
    # Product-IMGs
    productImg = productSoup.find_all('img', id="landingImage")
    if len(productImg) > 0:
        productImg = productImg[0]['data-a-dynamic-image']
        productImg = json.loads(productImg)
    
    # Product-Rating
    productRating = productSoup.find_all('span', class_="a-icon-alt")
    if len(productRating) > 0:
        productRating = productRating[0].get_text().strip()

    # Product-Stars
    productStars = productSoup.find_all('table', id="histogramTable")
    if len(productStars) > 0:
        productStars = productStars[0].get_text().replace('\n', '').split('%')
        temp = []
        for i in range(len(productStars)-1):
            temp.append(float(productStars[i][-2:]))
        productStars = temp
    
    # Product-Features
    productFeatures = productSoup.find_all('div', id='feature-bullets')
    if len(productFeatures) > 0:
        productFeatures = productFeatures[0].get_text().strip()
        productFeatures = re.split('\n|  ',productFeatures)
        temp = []
        for i in range(len(productFeatures)):
            if productFeatures[i]!='' and productFeatures[i]!=' ' :
                temp.append( productFeatures[i].strip() )
        productFeatures = temp
    
    # Product-Specs
    ids = { 'productDetails_techSpec_section_1' : 'table', 'detailBullets_feature_div' : 'div' }
    for key, value in ids.items():
        productSpecs = productSoup.find_all(value, id=key)
        if len(productSpecs) > 0:
            productSpecs = productSpecs[0].get_text().strip()
            productSpecs = re.split('\n|\u200e|  ',productSpecs) 
            temp = []
            for i in range(len(productSpecs)):
                if productSpecs[i]!='' and productSpecs[i]!=' ' :
                    temp.append( productSpecs[i].strip() )
            productSpecs = temp
            break
    
    # Product-Details
    ids = { 'productDetails_db_sections' : 'div' }
    for key, value in ids.items():
        productDetails = productSoup.find_all(value, id=key)
        if len(productDetails) > 0:
            productDetails = productDetails[0].get_text()
            productDetails = re.split('\n|  ',productDetails) 
            temp = []
            for i in range(len(productDetails)):
                if productDetails[i]!='' and productDetails[i]!=' ' :
                    temp.append( productDetails[i].strip() )
            productDetails = temp
            break
    
    context1 = ''
    for i in range(1, len(productFeatures)-1):
        context1 = context1 + 'Product has ' + productFeatures[i].replace(' | ', ', ') + '. '
    
    context2 = ''
    for i in range(0, len(productSpecs), 2):
        context2 = context2 + productSpecs[i] + ' is ' + productSpecs[i+1] + '. '
    
    details = {
        'product_data' : {
            'productNames' : productNames,
            'productDiscountPrice' : productDiscountPrice,
            'productActualPrice' : productActualPrice,
            'productRating' : productRating,
            'productStars' : productStars,
            'productImg' : productImg,
            'productFeatures' : productFeatures,
            'productSpecs' : productSpecs,
            'productDetails' : productDetails,
            'context1' : context1, 
            'context2' : context2 
        }
    }
    return details

## Helper Funtions
def find_answer(answer1, answer2):
    #print(answer1, type(answer1))
    answer1 = answer1.split(' ')
    answer2 = answer2.split(' ')
    answer = []

    for word in answer1:
        if not (word in answer):
            answer.append(word)
    answer.append(',')
    for word in answer2:
        if not (word in answer):
            answer.append(word)
    
    answer = " ".join(answer)
    answer = answer.strip().strip(',')
    return answer

def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list

## Function to Scrape product review data 
def scrape_reviews( reviewsURL ):
    headers_ = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    allReviewSoup_ = []
    pageNo = 0
    while 1 :
        ## Look into all pages untill the page is empty
        pageNo += 1
        reviewsPage_ = requests.get(reviewsURL+'&pageNumber='+str(pageNo), headers=headers_)
        reviewsSoup_ = bs(reviewsPage_.content,'html.parser')
        names = reviewsSoup_.find_all('span',class_='a-profile-name')
        if len(names)<=2 and pageNo>1:
            break
        else :
            allReviewSoup_.append(reviewsSoup_)
        
    ## get reviews from all pages
    cust_name = []
    review_title = []
    rate = []
    review_content = []
    for reviewsSoup in allReviewSoup_:
        names = reviewsSoup.find_all('span',class_='a-profile-name')
        for i in range(1,len(names)):
            cust_name.append(names[i].get_text())
        
        title = reviewsSoup.find_all('a',class_='review-title-content')
        
        for i in range(0,len(title)):
            review_title.append(title[i].get_text())
        review_title[:] = [titles.lstrip('\n') for titles in review_title]
        review_title[:] = [titles.rstrip('\n') for titles in review_title]

        rating = reviewsSoup.find_all('i',class_='review-rating')
        for i in range(0,len(rating)):
            rate.append(int(rating[i].get_text()[0]))
        
        review = reviewsSoup.find_all("span",{"data-hook":"review-body"})
        for i in range(0,len(review)):
            review_content.append(review[i].get_text())
        review_content[:] = [reviews.lstrip('\n') for reviews in review_content]
        review_content[:] = [reviews.rstrip('\n') for reviews in review_content]

    reviewDataset = pd.DataFrame()
    reviewDataset['Customer Name'] = cust_name
    reviewDataset['Review title']  = review_title
    reviewDataset['Ratings']       = rate
    reviewDataset['Reviews']       = review_content

    return [reviewDataset, review_content]


### Streamlit app
st.set_page_config(
    page_title="eSeller",
    page_icon="https://github.com/Aditya-R-Chakole/AQnA-System/blob/main/seller-png.png?raw=true",
    layout="wide",
)
st.markdown("""
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
""", unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #212121;">
  <a class="navbar-brand" href="https://share.streamlit.io/aditya-r-chakole/eSeller-QnA-System/main/app.py" target="_blank" style="font-size: 25px;"><b>eSeller</b> | Question-Answering and Sentiment Analysis System</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
</nav>

<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


## Initialize Variables
productURL = ''
productName = ''
productID = ''

## Sidebar Code
webPageSideBar = st.sidebar
webPageSideBar.markdown(f'''<h3><b>Amazon Product Link</b> </h3>''', unsafe_allow_html=True)
productURL_Box = webPageSideBar.empty()

productURL = productURL_Box.text_input('', placeholder="Enter Amazon product link here... ")
webPageSideBar.markdown(f'''<h4 style="display:flex; direction:row; justify-content: space-evenly;"> <b>OR</b> </h4>''', unsafe_allow_html=True)
if productURL != '':
    st.session_state['productURL'] = productURL

## Template Products
if webPageSideBar.button('Nokia C01 Plus 4G'):
    productURL = 'https://www.amazon.in/Nokia-C01-Screen-Selfie-Storage/dp/B09VCBGWFZ/ref=sr_1_99?crid=1VT9QAPKD4HF&keywords=mobile&qid=1661646921&sprefix=mobile%2Caps%2C218&sr=8-99'
    st.session_state['productURL'] = productURL
if webPageSideBar.button('Boult Audio  Oak in-Ear Earphones'):
    productURL = 'https://www.amazon.in/Boult-Audio-BassBuds-Oak-Earphones/dp/B091JF2TFD/ref=sr_1_34_sspa?crid=Y3DNK9KAGAFF&keywords=earphones&qid=1661645548&sprefix=earphones%2Caps%2C224&sr=8-34-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUFDOEFVVEszV01ZSFQmZW5jcnlwdGVkSWQ9QTAyMDA2NzcxS0hFTlRUVEs5VkpVJmVuY3J5cHRlZEFkSWQ9QTAxMTcwODgyT0k5STRRRVg3UUpTJndpZGdldE5hbWU9c3BfYXRmX25leHQmYWN0aW9uPWNsaWNrUmVkaXJlY3QmZG9Ob3RMb2dDbGljaz10cnVl'
    st.session_state['productURL'] = productURL
if webPageSideBar.button('Orient Electric Dry Iron'):
    productURL = 'https://www.amazon.in/dp/B07GTJD6C8/ref=s9_acsd_al_bw_c2_x_20_t?pf_rd_m=A1K21FY43GMZF8&pf_rd_s=merchandised-search-10&pf_rd_r=MHPFRTFQJ06J32SQSV1T&pf_rd_t=101&pf_rd_p=fdbbc7f9-2486-4343-bab8-5147d713b841&pf_rd_i=27916840031'
    st.session_state['productURL'] = productURL
if webPageSideBar.button('Fifine T730 USB Microphone Kit'):
    productURL = 'https://www.amazon.in/T730-Microphone-Capsule-Cardioid-Recording/dp/B095WF6PG5/ref=sr_1_1_sspa?keywords=fifine+t730+usb+microphone&qid=1661645313&sprefix=fifin+t730+%2Caps%2C282&sr=8-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzRFFVWk9ZUzJPUEEyJmVuY3J5cHRlZElkPUEwNTg3MDExMlpZMEE4NDhOWEo5OCZlbmNyeXB0ZWRBZElkPUEwNDIyNzc2M0JHUlBKTko5SE9PMCZ3aWRnZXROYW1lPXNwX2F0ZiZhY3Rpb249Y2xpY2tSZWRpcmVjdCZkb05vdExvZ0NsaWNrPXRydWU='
    st.session_state['productURL'] = productURL
if webPageSideBar.button('Amazon Brand Digital Mens Watch'):
    productURL = 'https://www.amazon.in/Amazon-Brand-Symbol-Analog-Digital-Watch-AZ-SYM-SS21-1155A/dp/B09HSPF81R/ref=sr_1_8?crid=3PFPV7MW3Z1BK&keywords=Amazon%2BBrand%2B-%2BSymbol%2Bdigital%2BMen%27s%2BWatch&qid=1661644270&sprefix=amazon%2Bbrand%2B-%2Bsymbol%2Bdigital%2Bmen%2Bs%2Bwatch%2B%2Caps%2C250&sr=8-8&th=1'
    st.session_state['productURL'] = productURL

if 'productURL' in st.session_state:
    productURL = st.session_state['productURL']
else :
    productURL = ''

if(productURL != ''):
    ## Scrape data and show the data 
    data = scrape_data(productURL)
    product_title = (data['product_data']['productNames']).split( '(' )
    title = (product_title[0]).split('with')
    st.markdown( f'<h3 style="color:#F7CA00;"> <b>{title[0]}</b> </h3>', unsafe_allow_html=True)
    if len(product_title) > 1:
        st.markdown( f'<h5 style="color:#ffffff;"> {"("+product_title[1]} </h5>', unsafe_allow_html=True)
    
    col0, col1 = st.columns([30, 70])
    ## Col 0
    with col0:
        imgURL = list(data['product_data']['productImg'].keys())
        st.markdown(f'''<img src={imgURL[0]} alt="product" width="100%" height="100%" style="display: flex; flex-direction:row; justify-content: space-evenly;">
                        <h6>   </h6>
                        <figcaption style="text-align: center;">{title[0]}</figcaption>''', unsafe_allow_html=True)

    ## Col 1
    with col1:
        MRP = float(re.sub('\D', '', data['product_data']['productActualPrice'][:-2]))
        OfferPrice = float(re.sub('\D', '', data['product_data']['productDiscountPrice'][:-2]))
        UP = sum(data['product_data']['productStars'][0:3])
        DOWN = sum(data['product_data']['productStars'][3:])

        rating = float(data['product_data']['productRating'][0:3])
        ratingHelper = None
        if rating>=0.5 and rating<1.5 :    
            ratingHelper = '<span class="fa fa-star checked"></span>'
        elif rating>=1.5 and rating<2.5 :    
            ratingHelper = '<span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span>'
        elif rating>=2.5 and rating<3.5 :    
            ratingHelper = '<span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span>'
        elif rating>=3.5 and rating<4.5 :
            ratingHelper = '<span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span>'
        else:
            ratingHelper = '<span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span>'
            
        st.markdown(f'''
        <div style="display: flex; flex-direction:row; justify-content: space-evenly;">
            <div class="card bg-dark text-white" style="width: 18rem;">
                <div class="card-body">
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#00C896;"><b>{data['product_data']['productDiscountPrice']}</b></h4>
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#B33F40;"><del>{data['product_data']['productActualPrice']}</del> <b style="color:#00C896;">🡇 {round((MRP-OfferPrice)*100/MRP, 1)}%</b> </h4>
                    <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#F7CA00"><b>Price</b></h5>
                </div>
            </div>
            <div class="card bg-dark text-white" style="width: 18rem;">
                <div class="card-body">
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#B33F40;"> <b style="color:#00C896;">🡅 {UP}%</b>  <b style="color:#B33F40;">🡇 {100-UP}%</b> </h4>
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFA41C;"> {ratingHelper} </h4>
                    <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#F7CA00"><b>Rating</b></h5>
                </div>
            </div>
        </div>''', unsafe_allow_html=True)

        ## sentiment analysis
        if 'like' not in st.session_state:
            reviews = scrape_reviews(productURL+'&reviewerType=all_reviews')
            MAX_FEATURES = 12000
            tokenizer = None
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            
            test_texts = tokenizer.texts_to_sequences(reviews[1])
            test_texts = pad_sequences(test_texts, maxlen=255)
            model = tensorflow.keras.models.load_model('sentimentAnalysisModel/')
            preds = model.predict(test_texts)
            preds = (1 * (preds >= 0.5))
            for i in range(len(preds)):
                if reviews[0]['Ratings'][i] > 3:
                    preds[i] = 1
            like  =  (sum(preds)/len(preds))*100
            st.session_state['test_texts'] = test_texts
            st.session_state['like'] = like

        st.markdown(f'''
        <div style="display: flex; flex-direction:row; justify-content: space-evenly;">
            <div class="card bg-dark text-white" style="width: 38rem; margin-top: 1rem; margin-bottom: 1rem;">
                <div class="card-body">
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#B33F40;"> <b style="color:#00C896;">🡅 {round(st.session_state['like'][0], 1)}%</b>  <b style="color:#B33F40;">🡇 {round(100.0-st.session_state['like'][0], 1)}%</b> </h4>
                    <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#F7CA00"><b>Sentiment Analysis results of all {len(st.session_state['test_texts'])-(len(st.session_state['test_texts'])%10)}+ Reviews</b></h5>
                </div>
            </div>
        </div>''', unsafe_allow_html=True)
            
        ## Q-A system
        question = st.text_input('', placeholder="Ask Anything ")
        if question != '':   
            answer1 = qna_bert(data['product_data']['context1'], question)
            answer2 = qna_bert(data['product_data']['context2'], question)
            answer = find_answer(answer1['answer'], answer2['answer'])
            st.success(answer)
else:
    st.markdown(f'''
        <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFFFFF;"><b>Welcome to <span style="color:#F7CA00;"><a href="https://share.streamlit.io/aditya-r-chakole/eseller/main/app.py" style="color: inherit;">eSeller</a></span></b></h4>
        <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFFFFF;"><b>This <span style="color:#F7CA00;">short tutorial</span> will walk you through all of the features of this application.</b></h5>
        <div style="display: flex; flex-direction:row; justify-content: space-evenly; padding:10px;">
            <div class="card bg-dark text-white" style="width: 75rem;">
                <div class="card-body">                                
                    <div style="padding:1px;">
                        <img src="https://github.com/Aditya-R-Chakole/eSeller-QnA-System/blob/main/startPage1.png?raw=true" alt="eSeller" width="100%" height="100%"></img>
                        <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFFFFF; position:absolute; top:60%; left:6.5%; width:325px;"><b>Select a <span style="color:#F7CA00;">Ecommerce Website</span> and put the <span style="color:#F7CA00;">Product Link</span> here.</b></h5>
                    </div>
                </div>
            </div>
        </div>
        <div style="display: flex; flex-direction:row; justify-content: space-evenly; padding:10px;">
            <div class="card bg-dark text-white" style="width: 75rem;">
                <div class="card-body">                                
                    <div style="padding:1px;">
                        <img src="https://github.com/Aditya-R-Chakole/eSeller-QnA-System/blob/main/startPage2.png?raw=true" alt="eSeller" width="100%" height="100%"></img>
                        <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFFFFF; position:absolute; top:76%; left:20%; width:300px;"><b>Verify the Product and <span style="color:#F7CA00;">Ask Any Product related question here</span>.</b></h5>
                    </div>
                </div>
            </div>
        </div>
        <div style="display: flex; flex-direction:row; justify-content: space-evenly; padding:10px;">
            <div class="card bg-dark text-white" style="width: 75rem;">
                <div class="card-body">                                
                    <div style="padding:1px;">
                        <img src="https://github.com/Aditya-R-Chakole/eSeller-QnA-System/blob/main/startPage3.png?raw=true" alt="eSeller" width="100%" height="100%"></img>
                        <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFFFFF; position:absolute; top:75%; left:15%; width:300px;"><b>Get the <span style="color:#F7CA00;">answer</span>, or else <span style="color:#F7CA00;">try changing the key word</span>.</b></h5>
                    </div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)