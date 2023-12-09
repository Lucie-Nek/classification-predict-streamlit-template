"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Images
from PIL import Image
import pickle

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Vectorizer
news_vectorizer = open("resources/Models/tfidf_vect_model.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/Files/train.csv")

# The main function where we will build the actual app
def main():
	"""Climate Change Twitter Sentiment Analysis App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("ECASA")
	st.subheader("EcoSentiment Analytics Sentiment Analysis")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Explore", "Feature Engineering", "Prediction", "About Us", "Contact Us"]
	selection = st.sidebar.radio("Choose Option", options)

	# Building out the "Home" page
	if selection == "Home":
		image = Image.open('resources/climate_change.png')
		st.image(image, caption='Climate Change')

		st.markdown("### Analysing Today for a Sustainable Tomorrow!")
		st.write("The ECASA App is based on Machine Learning models that can classify whether user sentiments on climate change are pro, anti, news-related, or neutral-based on their tweets. Our planet is bleeding and people care. Because businesses care about what is important to its people, ECASA is a step in the right direction of saving the planet and putting people first.")

	# Building out the "Explore" page
	if selection == "Explore":
		st.markdown("### Exploratory Data Analysis (EDA)")
		# You can read a markdown file from supporting resources folder
		st.markdown("This section contains insights on the loaded dataset and its output")

		# Display the unprocessed data
		st.markdown("##### Raw Twitter data")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		option = st.sidebar.selectbox('Select visualization', ('Common words', 'Popular hashtags', 'Top entities'))

		if st.checkbox('Show visualizations'):
			if option == 'Common words':
				image = Image.open("resources/Visuals/word_cloud_most_common_words.png")
				st.image(image)
			else:
				with st.expander('Pro Sentiments'):
					image = Image.open('resources/Visuals/word_cloud_most_common_words_pro.png')
					st.image(image)
				with st.expander('News'):
					image = Image.open('resources/Visuals/word_cloud_most_common_words_news.png')
					st.image(image)
				with st.expander('Neutral Sentiments'):
					image = Image.open('resources/Visuals/word_cloud_most_common_words_neutral.png')
					st.image(image)
				with st.expander('Anti Sentiments'):
					image = Image.open('resources/Visuals/word_cloud_most_common_words_anti.png')
					st.image(image)
					
			if option == 'Popular hashtags':
				image = Image.open('resources/Visuals/most_common_hashtags.png')
				st.image(image)
			else:
				with st.expander('Pro Sentiments'):
					image = Image.open('resources/Visuals/most_common_hashtags_pro.png')
					st.image(image)
				with st.expander('News'):
					image = Image.open('resources/Visuals/most_common_hashtags_news.png')
					st.image(image)
				with st.expander('Neutral Sentiments'):
					image = Image.open('resources/Visuals/most_common_hashtags_neutral.png')
					st.image(image)
				with st.expander('Anti Sentiments'):
					image = Image.open('resources/Visuals/most_common_hashtags_anti.png')
					st.image(image)
					
			if option == 'Top entities':
				image = Image.open('resources/Visuals/top_entities.png')
				st.image(image)
			else:
				with st.expander('Pro Sentiments'):
					image = Image.open('resources/Visuals/top_enities_pro.png')
					st.image(image)
				with st.expander('News'):
					image = Image.open('resources/Visuals/top_entities_news.png')
					st.image(image)
				with st.expander('Neutral Sentiments'):
					image = Image.open('resources/Visuals/top_entities_neutral.png')
					st.image(image)
				with st.expander('Anti Sentiments'):
					image = Image.open('resources/Visuals/top_entities_anti.png')
					st.image(image)

	# Building out the "Feature Engineering" page
	if selection == "Feature Engineering":
		st.markdown("### Feature Engineering")
		# You can read a markdown file from supporting resources folder
		st.markdown("This section contains insights on how features were processed for machine learning")

		# Display the unprocessed data
		st.markdown("##### Balancing of data")
		if st.checkbox('Show unbalanced data'): # data is hidden if box is unchecked
			image = Image.open('resources/Visuals/sentiment_distribution.png')
			st.image(image)

		if st.checkbox('Show balanced data'): # data is hidden if box is unchecked
			image = Image.open('resources/Visuals/balanced_distribution.png')
			st.image(image)

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with Classifier Models")
		option = st.sidebar.selectbox(
            'Select the model from the Dropdown',
            ('Logistic Regression', 'Support Vector Classifier', 'Naive Bayes Classifier'))
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		# model selection options
		if option == 'Logistic Regression':
			model = "resources/Models/lr_model.pkl"
		elif option == 'Support Vector Classifier':
			model = "resources/Models/svc_model.pkl"
		elif option == 'Naive Bayes Classifier':
			model = "resources/Models/nb_model.pkl"

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(model),"rb"))
			prediction = predictor.predict(vect_text)

			word = ''
			if prediction == 0:
				word = '"**Neutral**". The tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				word = '"**Pro**". The tweet supports the belief of man-made climate change'
			elif prediction == 2:
				word = '**News**. The tweet links to factual news about climate change'
			else:
				word = '**Anti**. The tweet does not believe in man-made climate change'

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(word))

	# Building out the About Us page
	if selection == "About Us":
		st.info("EcoSentiment Analytics - Empowering Sustainability Through Data")
		st.write("EcoSentiment Analytics is at the forefront of a new era where environmental awareness meets cutting-edge data analytics. Our mission is to harness the power of data-driven insights to cultivate a sustainable future. We are dedicated to developing cutting-edge analytics solutions that empower individuals, businesses, and policymakers with the knowledge needed to make informed decisions for the betterment of the environment.")
		
		st.info("Our Vision:")
		st.write("To be the leading global data-driven solutions provider")

		st.info("Meet the team")
		Fhulu = Image.open('resources/Team_pics/Fhulu.jpg')
		Fhulu1 = Fhulu.resize((150, 155))
		Jonathan = Image.open('resources/Team_pics/Jonathan.jpg')
		Jonathan1 = Jonathan.resize((150, 155))
		Mulalo = Image.open('resources/Team_pics/Mulalo.jpg')
		Mulalo1 = Mulalo.resize((150, 155))
		Mkhosi = Image.open('resources/Team_pics/Mkhosi.jpg')
		Mkhosi1= Mkhosi.resize((150, 155))
		Amanda = Image.open('resources/Team_pics/Amanda.jpg')
		Amanda1 = Amanda.resize((150, 155))
		Lucie = Image.open('resources/Team_pics/Lucie.jpg')
		Lucie1 = Lucie.resize((150, 155))

		col1, col2, col3 = st.columns(3)
		with col1:
			st.image(Fhulu1, width=150, caption="Fhulufhelo: Team Lead")
		with col3:
			st.image(Mulalo1, width=150, caption="Mulalo: Data Scientist")
		with col2:
			st.image(Jonathan1, width=150, caption="Jonathan: Project Manager")
		col1, col2, col3= st.columns(3)
		with col1:
			st.image(Mkhosi1, width=150, caption="Mkhosi: Senior Data Analyst")
		with col2:
			st.image(Amanda1, width=150, caption="Amanda: Data Analyst")
		with col3:
			st.image(Lucie1, width=150, caption="Lucie: ML Engineer")

	# Build the Contact us page
	if selection == "Contact Us":
		image = Image.open('resources/contactus.png')
		st.image(image)
		
		col1, col2 = st.columns(2)
		with col1:
			st.subheader("Contact info")
			st.write("123, Greenway Street")
			st.write("Johannesburg, 2052, South Africa")
			st.write("Telephone:+27-11-456-7890")
			st.write("Email: ecasa@ecoanalytics.com")
			
		with col2:
			st.subheader("Send Us")
			email = st.text_input("Enter your email")
			message = st.text_area("Enter your message")
			st.button("Send")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
