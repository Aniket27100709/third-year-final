# Core pkgs
# where all the features will be inside
import streamlit.components.v1 as components
import sweetviz as sv
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
from wordcloud import WordCloud
# EDA pkgs

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import codecs  # will help to load our files
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
# natural language processing
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import string
import nltk
nltk.download('stopwords')
ps = PorterStemmer()


data2="sales.csv"
# component pkgs


def st_display_sweetviz(report_html, width=1000, height=500):
    report_file = codecs.open(report_html, 'r')
    page = report_file.read()
    components.html(page, width=width, height=height, scrolling=True)


def main():
    """A simple EDA app with streamlit components"""
    menu = ["Home","Product's Analysis","Product's Sentiment Analysis",
            "Pandas Profile", "Sweetviz Report", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Pandas Profile":
        st.subheader("Automated EDA with Pandas Profile")
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if data_file is not None:
            df = pd.read_csv(data_file, encoding="unicode_escape")
            st.dataframe(df.head())
            profile = ProfileReport(df)
            st_profile_report(profile)
    elif choice=="WordCloud":
        data=pd.read_csv('amazon(1).csv')
        text=data['customer_reviews']
        word_cloud=WordCloud().generate(text)
        plt.imshow(word_cloud)
        st.pyplot()


    elif choice == "Product's Sentiment Analysis":
        def transform_text(text):
            text = text.lower()
            text = nltk.word_tokenize(text)
            y = []
            for i in text:
                if i.isalnum():
                    y.append(i)
            text = y[:]
            y.clear()
            for i in text:
                if i not in stopwords.words('english') and i not in string.punctuation:
                    y.append(i)
            text = y[:]
            y.clear()
            for i in text:
                y.append(ps.stem(i))

            return " ".join(y)

        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))

        st.title("Product's Sentiment Analysis")
        inputs = st.text_input("Enter Product's Name")

        if st.button('Output'):
            transformed = transform_text(inputs)
            # vectorizing

            vec = tfidf.transform([transformed])

            result = model.predict(vec)[0]
            if result == 1:
                st.header("Positive")
            else:
                st.header("Negative")

    elif choice == "Sweetviz Report":
        st.subheader("Automated EDA with Sweetviz")
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if data_file is not None:
            df = pd.read_csv(data_file, encoding="unicode_escape")
            st.dataframe(df.head())
            if st.button("Generate Sweetviz Report"):
               # Normal workflow
                report = sv.analyze(df)
                report.show_html()
                st_display_sweetviz("SWEETVIZ_REPORT.html")

    elif choice == "About":
        st.title("About APP")
        st.subheader(
            'This app actually give detailed analysis of products which are necessary for the businessman and investors to get deal in. A detailed is being generated where products are compared and the most frequent products which are purchased have more positive reviews')
        # components.iframe("https://google.com")

    elif choice=="Product's Analysis":

        st.title("Product's Analysis")
        df=pd.read_csv(data2,encoding="unicode_escape")
        st.sidebar.header("Enter Product details:")
        product=st.sidebar.multiselect("Select the Product",options=df['Product'].unique(),default=df['Product'].unique())
        customer = st.sidebar.multiselect("Select the type of customer", options=df['Customer'].unique(),
                                         default=df['Customer'].unique())
        gender = st.sidebar.multiselect("Select the Gender", options=df['Gender'].unique(),
                                          default=df['Gender'].unique())
        df_selection=df.query("Product==@product & Customer==@customer & Gender==@gender")
        available=int(df_selection["Quantity"].sum())
        average_rating=round(df_selection["Rating"].mean(),1)
        star_rating=":star:" * int(round(average_rating,0))
        average_available=round(df_selection["Quantity"].mean(),2)
        l,m,r=st.columns(3)
        with l:
            st.subheader("No. of Product's available")
            st.subheader(f"{available}")
        with r:
            st.subheader("Rating for the products")
            st.subheader(f"{average_rating} {star_rating}")
        sales=(df_selection.groupby(by=["Sales"]).sum()[["Product"]].sort_values(by="Product"))
        fig_sales=px.bar(sales,x="Product",y=Sales.index,orientation="h",title="<b>No of reviews of given product</b>",color_discrete_sequence=["#008380"] * len(sales))
        st.plotly_chart(sales)
        st.dataframe(df_selection.head())

    else:
        st.title("Home")
        #components.html("<p style='color:red;'> Streamlit components is awsome </p>")
        st.header('Customer Requirement Analysis based on User Generated Content')
        st.image('wavy.jpg')


if __name__ == '__main__':
    main()
