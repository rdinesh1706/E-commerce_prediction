from nltk import ne_chunk, pos_tag
from rake_nltk import Rake
from collections import Counter
import nltk
import cv2
import pytesseract
import sklearn
from PIL import Image
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import easyocr
import io
from scipy.signal import fftconvolve
from scipy.signal import wiener
from scipy.signal import deconvolve
from scipy.signal import convolve2d
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk import FreqDist
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')


def de_blur(image):
    pass


# st.write("hello Worldd")
def model():
    # st.set_page_config(layout="wide")
    # st.markdown(""" <div style='text-align:center; font-family:"Times New Roman";'>
    #     <h1 style='color:#0000ff;'>Sales Conversion Classifier</h1>
    # </div>""", unsafe_allow_html=True)

    # loading the mapped dictionaries for label encoder
    with open('mapping_dict1.pkl', 'rb') as file:
        mapping_dict_channelGrouping = pickle.load(file)

    with open('mapping_dict1.pkl', 'rb') as file:
        mapping_dict_device_browser = pickle.load(file)

    with open('mapping_dict3.pkl', 'rb') as file:
        mapping_dict_device_operatingSystem = pickle.load(file)

    with open('mapping_dict4.pkl', 'rb') as file:
        mapping_dict_Products = pickle.load(file)

    with open('mapping_dict5.pkl', 'rb') as file:
        mapping_dict_region = pickle.load(file)
    # loading csv for visualizations
    dfi = pd.read_csv('dff.csv')
    x_test = pd.read_csv('x_test.csv')
    y_test = pd.read_csv('y_test.csv')
    # # loading scaler and model
    scaler = joblib.load('standard_scaler.pkl')

    model = joblib.load('dtree.pkl')

    model1 = joblib.load('svm_model.pkl')
    model2 = joblib.load('random_forest_model.pkl')

    st.write("hello")

    tab2, tab3, tab4 = st.tabs(
        ['Predict', 'EDA', 'tab'])

    # setting tab and dropdown for feature selection
    with tab2:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap='small')

        with col1:
            count_session = st.number_input("Enter the Count")

        with col2:
            count_hit = st.number_input("Enter the Hit Count")

        with col3:
            num_interactions = st.number_input(
                "Enter the number of interactions")

        with col4:
            channel_Grouping = st.selectbox(
                "Channel Grouping", options=list(mapping_dict_channelGrouping.keys()))

        col5, col6, col7, col8 = st.columns([1, 1, 1, 1], gap='small')

        with col5:
            device_browser = st.selectbox(
                'Device Browser', options=list(mapping_dict_device_browser.keys()))

        with col6:
            device_operatingSystem = st.selectbox(
                "Operating System", options=list(mapping_dict_device_operatingSystem.keys()))

        with col7:
            Products = st.selectbox(
                'Select Products', options=list(mapping_dict_Products.keys()))

        with col8:
            region = st.selectbox(
                "Select Region", options=list(mapping_dict_region.keys()))

    # creating the dataframe after user inputs
            df = pd.DataFrame({
                'count_session': [float(count_session)],
                'count_hit': [float(count_hit)],
                'num_interactions': [float(num_interactions)],
                'channelGrouping': [channel_Grouping],
                'device_browser': [device_browser],
                'device_operatingSystem': [device_operatingSystem],
                'Products': [Products],
                'region': [region]})

    # creating a dataframe with mapped features

        df_encoded = df.copy()
        df_encoded['channelGrouping'] = df['channelGrouping'].map(
            mapping_dict_channelGrouping)
        df_encoded['device_browser'] = df['device_browser'].map(
            mapping_dict_device_browser)
        df_encoded['device_operatingSystem'] = df['device_operatingSystem'].map(
            mapping_dict_device_operatingSystem)
        df_encoded['Products'] = df['Products'].map(mapping_dict_Products)
        df_encoded['region'] = df['region'].map(mapping_dict_region)

        dfc = df_encoded.drop(
            ['count_session', 'count_hit', 'num_interactions'], axis=1)

    # select numeric columns for scaling

        dfn = df_encoded.drop(['channelGrouping', 'device_browser',
                               'device_operatingSystem', 'Products', 'region'], axis=1)

    # select numeric columns

        dfn_scaled = pd.DataFrame(scaler.transform(dfn), columns=dfn.columns)

        dff = pd.concat([dfn_scaled, dfc], axis=1)

        if st.button('Predict'):
            prediction = model.predict(dff)

            if prediction == 1:
                st.success("Sale converted")
            else:
                st.warning('Sale Not Converted')

    # creating a about section in tab3

    # creating a home tab with information about the model and the project

    with tab3:
        # visualization for the eda analysis
        st.markdown("EDA of the data are given below")
        col1, col2 = st.columns([1, 1], gap='small')
        # pairplot
        with col1:
            st.title('Pair plot')
            fig = sns.pairplot(dfi, hue='has_converted')
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.title('Correlation Heatmap')
            import plotly.express as px
            corrmat = dfi.corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(corrmat, annot=True, cmap='coolwarm',
                        fmt='.2f', linewidths=0.5)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        col3, col4 = st.columns([1, 1], gap='small')
        # boxplot
        with col3:
            st.title('Box Plot')
            st.set_option('deprecation.showPyplotGlobalUse',
                          False)  # Disable matplotlib warning

            sns.set(style="whitegrid")
            plt.figure(figsize=(15, 12))

            # List of variables for which you want to create box plots
            variables = ['count_session', 'count_hit', 'num_interactions', 'channelGrouping',
                         'device_browser', 'device_operatingSystem', 'Products', 'region']

            # create a collage of box plots
            for i, variable in enumerate(variables, start=1):
                plt.subplot(4, 2, i)
                sns.boxplot(x='has_converted', y=variable, data=dfi)
                plt.title(f'Box Plot for {variable}')
                plt.xlabel('Has Converted')
                plt.ylabel(variable)

            # adjust layout
            plt.tight_layout()

            # Display the box plots using st.pyplot
            st.pyplot(plt.gcf(), use_container_width=True)
        # 3d scatter plot
        with col4:
            from mpl_toolkits.mplot3d import Axes3D
            st.title('3D Scatter Plot')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # scatter plot
            ax.scatter(dfi['count_session'], dfi['count_hit'],
                       dfi['num_interactions'])

            # customize the plot if needed
            ax.set_xlabel('Count Session')
            ax.set_ylabel('Count Hit')
            ax.set_zlabel('Num Interactions')
            ax.set_title('3D Scatter Plot')

            st.pyplot(fig, use_container_width=True)

    with tab4:
        # using different models
        col1, col2, col3 = st.columns([1, 1, 1], gap='small')
        with col1:
            st.title("SVM Model")
            with st.container():
                st.markdown(
                    ''' 
                    <div style="background-color: red; padding: 15px; border-radius: 10px;">
                            <p style="font-size: 18px;">Support Vector Machine (SVM) Model Overview:</p>
                            <p style="font-size: 18px;">A Support Vector Machine is a powerful supervised machine learning algorithm used for both classification and regression tasks.</p>
                            <p style="font-size: 18px;">Key Features:</p>
                            <ul style="font-size: 18px;">
                                <li>Effective for both linear and non-linear data separation.</li>
                                <li>Maximizes the margin between classes for robust generalization.</li>
                                <li>Works well in high-dimensional spaces.</li>
                                <li>Kernel trick allows mapping input features into higher-dimensional space.</li>
                                <li>Sensitive to feature scaling.</li>
                            </ul>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        with col2:
            st.title('Random Forest Model')
            with st.container():
                st.markdown(
                    '''
                    <div style="background-color: red; padding: 15px; border-radius: 10px;">
                        <p style="font-size: 18px;">Random Forest Model Overview:</p>
                        <p style="font-size: 18px;">Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of individual trees.</p>
                        <p style="font-size: 18px;">Key Features:</p>
                        <ul style="font-size: 18px;">
                            <li>Ensemble of decision trees for improved accuracy and generalization.</li>
                            <li>Handles missing values and maintains accuracy even with a large number of features.</li>
                            <li>Reduces overfitting compared to individual decision trees.</li>
                            <li>Can handle both categorical and numerical data.</li>
                            <li>Random feature selection and bootstrapped sampling enhance diversity.</li>
                        </ul>
                    </div>
                    ''', unsafe_allow_html=True
                )

        with col3:
            st.title('Decision Tree Model')
            with st.container():
                st.markdown(
                    '''
                    <div style="background-color: red; padding: 15px; border-radius: 10px;">
                        <p style="font-size: 18px;">Decision Tree Model Overview:</p>
                        <p style="font-size: 18px;">A Decision Tree is a popular machine learning algorithm used for both classification and regression tasks. It builds a tree-like structure by recursively splitting the dataset based on the most significant features.</p>
                        <p style="font-size: 18px;">Key Features:</p>
                        <ul style="font-size: 18px;">
                            <li>Simple yet powerful for interpreting and visualizing decisions.</li>
                            <li>Handles both numerical and categorical data.</li>
                            <li>Non-parametric and requires minimal data preprocessing.</li>
                            <li>Prone to overfitting, but techniques like pruning help mitigate it.</li>
                            <li>Efficient for binary and multi-class classification problems.</li>
                        </ul>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        col1, col2, col3 = st.columns([1, 1, 1], gap='Small')
        with col1:
            # using svm model

            if st.button('Predict (SVM model)'):

                prediction1 = model1.predict(dff)

                if prediction1 == 1:
                    st.success('Sale Converted')
                else:
                    st.warning('Sale Not Converted')

        with col2:
            # using random forest model

            if st.button('Predict (Decision Tree Model)'):

                prediction = model.predict(dff)

                if prediction == 1:
                    st.success('Sale Converted')
                else:
                    st.warning('Sale Not Converted')

        # col1, col2, col3 = st.columns([1, 1, 1], gap='small')
        # with col1:
        #     with st.container():
        #         st.markdown(
        #             '''
        #             <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
        #                 <p style="font-size: 18px;">Support Vector Machine (SVM) Model Performance Metrics:</p>
        #                 <p style="font-size: 18px;">The following metrics provide insights into the effectiveness of the SVM model:</p>
        #                 <ul style="font-size: 18px;">
        #                     <li>F1 Score: 0.7655</li>
        #                     <li>Accuracy: 0.7731</li>
        #                     <li>Precision: 0.7838</li>
        #                     <li>Recall: 0.7731</li>
        #                 </ul>
        #                 <p style="font-size: 18px;">These metrics illustrate the SVM model's performance in terms of precision, recall, and accuracy in classifying instances.</p>
        #             </div>
        #             ''', unsafe_allow_html=True
        #         )

        # with col2:
        #     with st.container():
        #         st.markdown(
        #             '''
        #             <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
        #                 <p style="font-size: 18px;">Random Forest Model Performance Metrics:</p>
        #                 <p style="font-size: 18px;">The following metrics provide insights into the effectiveness of the Random Forest model:</p>
        #                 <ul style="font-size: 18px;">
        #                     <li>F1 Score: 0.8094</li>
        #                     <li>Accuracy: 0.8105</li>
        #                     <li>Precision: 0.8104</li>
        #                     <li>Recall: 0.8105</li>
        #                 </ul>
        #                 <p style="font-size: 18px;">These metrics demonstrate the Random Forest model's ability to balance precision, recall, and accuracy in classifying instances.</p>
        #             </div>
        #             ''', unsafe_allow_html=True
        #         )

        # with col3:
        #     with st.container():
        #         st.markdown(
        #             '''
        #             <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
        #                 <p style="font-size: 18px;">Decision Tree Model Performance Metrics:</p>
        #                 <p style="font-size: 18px;">The following metrics provide insights into the effectiveness of the Decision Tree model:</p>
        #                 <ul style="font-size: 18px;">
        #                     <li>F1 Score: 0.8216</li>
        #                     <li>Accuracy: 0.8225</li>
        #                     <li>Precision: 0.8222</li>
        #                     <li>Recall: 0.8225</li>
        #                 </ul>
        #                 <p style="font-size: 18px;">These metrics indicate the Decision Tree model's ability to balance precision, recall, and accuracy in classifying instances.</p>
        #             </div>
        #             '''
        #         )


def perform_ocr(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    text = '\n'.join([res[1] for res in result])
    return text


def sharpen_image(image, strength=1):
    kernel = np.array([[-1, -1, -1], [-1, 9+strength, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def detect_edges(image_path):
    # Read the image
    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


def blur(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image


def de_blurr(image):
    def create_gaussian_kernel(kernel_size, sigma):
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = np.exp(-((i - center) **
                                      2 + (j - center)**2) / (2 * sigma**2))

        kernel /= np.sum(kernel)  # Normalize the kernel to sum to 1
        return kernel

        # Example usage
    kernel_size = 5
    sigma = 1.5
    kernel = create_gaussian_kernel(kernel_size, sigma)
    deblurred_image = wiener(image, kernel)
    return deblurred_image


def filter():
    st.title("Image to Text Conversion using EasyOCR")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if uploaded_file is not None:
        # sharp

        imagesharp = np.array(Image.open(uploaded_file).convert('RGB'))
        # apply sharpening filter
        sharpened_image = sharpen_image(imagesharp)
        # Display Sharp image
        st.image(sharpened_image, caption='Sharp Image', use_column_width=True)

    # edge
        imagesharp = np.array(Image.open(uploaded_file).convert('RGB'))
        # apply sharpening filter
        edge_image = detect_edges(imagesharp)
        st.image(edge_image, caption='edge_image Image', use_column_width=True)

    # blur
        imageedge = np.array(Image.open(uploaded_file).convert('RGB'))
        # apply sharpening filter
        blur_image = blur(imageedge)
        st.image(blur_image, caption='blur_image Image',
                 use_column_width=True)

    # de-blur
        imagedeedge = np.array(Image.open(uploaded_file).convert('RGB'))
        # apply sharpening filter
        blur_image = de_blur(imagedeedge)
        st.image(image, caption='deblur Image', use_column_width=True)

        # Read the image
        image = Image.open(uploaded_file)

        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # Check if the uploaded image contains text
        try:
            ocr_result = perform_ocr(img_bytes)
            if ocr_result:
                st.write("OCR Result:")
                st.write(ocr_result)
            else:
                st.write("No text detected in the image.")
        except Exception as e:
            st.write("OCR could not be performed. Error:", e)


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# def plot_word_frequency(text):
#     words = nltk.tokenize.word_tokenize(text)
#     word_freq = Counter(words)
#     word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[
#                      :10])  # Selecting only the first 10 items
#     fig, ax = plt.subplots()
#     ax.bar(word_freq.keys(), word_freq.values())
#     ax.set_xlabel('Words')
#     ax.set_ylabel('Frequency')
#     ax.set_title('Word Frequency Bar Chart')
#     plt.xticks(rotation=90, ha='right')
#     st.pyplot(fig)

def stem_words(words):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words


def plot_word_frequency(text):
    words = word_tokenize(text)
    stemmed_words = stem_words(words)
    word_freq = Counter(stemmed_words)
    word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[
                     :10])  # Selecting only the first 10 items
    fig, ax = plt.subplots()
    ax.bar(word_freq.keys(), word_freq.values())
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Word Frequency Bar Chart')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)


def extract_named_entities(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    named_entities = ne_chunk(tagged_words)
    named_entities = [
        chunk for chunk in named_entities if hasattr(chunk, 'label')]
    named_entities = [' '.join([c[0] for c in chunk])
                      for chunk in named_entities]
    return named_entities


def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()


def text():
    st.title("NLP Analysis Tool")
    sentence = st.text_input("Enter a sentence:")

    if sentence:
        st.subheader("Word Frequency Analysis")
        plot_word_frequency(sentence)

        st.subheader("Named Entity Recognition")
        entities = extract_named_entities(sentence)
        st.write(entities)

        st.subheader("Keywords Extraction")
        keywords_list = extract_keywords(sentence)
        st.write(keywords_list)


data = {
    'product_id': ['Product1', 'Product2', 'Product3', 'Product4', 'Product5'],
    'smartphone': ['Apple iPhone', 'Samsung Galaxy', 'Google Pixel', 'OnePlus Nord', 'Xiaomi Redmi'],
    'fruits': ['apple', 'orange', 'pineapple', 'banana', 'liche'],
    "cloths": ['Shirt', 'Pant', 'Jean', 'Saree', 'Chudithar']
}

# Create DataFrame
df = pd.DataFrame(data)


def generate_recommendations(user_input, df, n=5):
    # TF-IDF Vectorization
    if user_input in df['fruits'].values:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['fruits'])

        # Calculate cosine similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Get index of product
        idx = df[df['fruits'] == user_input].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort products based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N similar products
        top_n = sim_scores[1:n+1]

        # Get recommended product names
        recommended_products = [df.iloc[idx]['fruits'] for idx, _ in top_n]
        return recommended_products
    elif user_input in df['smartphone'].values:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['smartphone'])

        # Calculate cosine similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Get index of product
        idx = df[df['smartphone'] == user_input].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort products based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N similar products
        top_n = sim_scores[1:n+1]

        # Get recommended product names
        recommended_products = [df.iloc[idx]['smartphone'] for idx, _ in top_n]
        return recommended_products
    elif user_input in df['cloths'].values:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['cloths'])

        # Calculate cosine similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Get index of product
        idx = df[df['cloths'] == user_input].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort products based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N similar products
        top_n = sim_scores[1:n+1]

        # Get recommended product names
        recommended_products = [df.iloc[idx]['cloths'] for idx, _ in top_n]
        return recommended_products
    # user_input not in df['fruits'].values:
    else:
        return ["Product not found in the dataset. Please try another product."]
    # lol = 'fruits'


def recom():
    st.title('Product Recommendation System')

    user_input = st.text_input('Enter a product name:')
    if st.button('Get Recommendation'):
        recommendations = generate_recommendations(user_input, df)
        st.write('Recommended Products:')
        for product in recommendations:
            st.write(product)


with st.sidebar:
    st.title("Final Project")
    show_table = st.radio(
        "Press the button for the view 👇",
        ["Model Prediction", "Image Processing", "Text", "Recommendation"],
    )


if show_table == "Model Prediction":
    model()
elif show_table == "Image Processing":
    filter()
elif show_table == "Text":
    text()
elif show_table == "Recommendation":
    recom()
