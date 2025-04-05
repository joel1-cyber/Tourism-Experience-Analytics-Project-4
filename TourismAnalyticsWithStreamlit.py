import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import streamlit as st
import os
from scipy.stats import chi2_contingency
import numpy as np
import plotly.express as px
import joblib
from category_encoders import TargetEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def MainDataset():
    #User Details
    userdf=pd.read_excel('User.xlsx')
    #converted city float to int
    userdf['CityId']=userdf['CityId'].fillna(0)
    #print(userdf.isnull().sum())   --To check nullvalues
    userdf['CityId']=userdf['CityId'].astype(int)
    # print(userdf.dtypes)
    # print(userdf.head())

    # Continent Details
    continentdf=pd.read_excel('Continent.xlsx')
    #print(continentdf.head())

    #merging both user and continent
    user_contitentdf=userdf.merge(continentdf,on='ContinentId',how='left')
    #print(user_contitentdf.head())

    # Region Details
    Regiondf=pd.read_excel('Region.xlsx')
    #print(Regiondf.head())
    #Merging user Details with Region
    user_continent_regiondf=user_contitentdf.merge(Regiondf,on='RegionId',how='left')
    #print(user_continent_regiondf.head())


    #Contry Details
    countrydf=pd.read_excel('Country.xlsx')
    #print(countrydf.head())
    user_continent_region_Countrydf=user_continent_regiondf.merge(countrydf,on='CountryId',how='left')
    #print(user_continent_region_Countrydf.head())

    #cityDetails
    citydf=pd.read_excel('City.xlsx')
    #print(citydf.head())
    user_continent_region_Country_citydf=user_continent_region_Countrydf.merge(citydf,on='CityId',how='left')
    # print(user_continent_region_Country_citydf.head())

    #------------------------------------------------------------------------------------------------------------------#
    #User Details Ready
    user_detailsdf=user_continent_region_Country_citydf.drop(columns=["ContinentId_x","RegionId_x","CountryId_x","CityId","RegionId_y","CountryId_y","ContinentId_y"])
    print("UserDetails : After Merging")
    print(user_detailsdf.head())
    # print("Null checking")
    # print(user_detailsdf.isnull().sum())

    #------------------------------------------------------------------------------------------------------------------#

    print("Attraction Details")
    Attractiondf=pd.read_excel('Updated_Item.xlsx')
    #print(Attractiondf.head())

    #Attraction City 
    Attraction_citydf=Attractiondf.merge(citydf,left_on='AttractionCityId',right_on='CityId',how='left')
    Attraction_citydf=Attraction_citydf.rename(columns={'CityName':'AttractedCity'})
    print(Attraction_citydf.head())

    #Attraction Coutry
    Attraction_city_countrydf=Attraction_citydf.merge(countrydf,on='CountryId',how='left')
    print('Country')
    Attraction_city_countrydf=Attraction_city_countrydf.rename(columns={'Country':'AttractedCountry'})


    print(Attraction_city_countrydf.head())


    #Attracted Region
    Attraction_city_country_regiondf=Attraction_city_countrydf.merge(Regiondf,on='RegionId',how='left')
    print('Region....................')
    Attraction_city_country_regiondf=Attraction_city_country_regiondf.rename(columns={'Region':'AttractedRegion'})
    print(Attraction_city_country_regiondf.head())

    #Attracted Continent
    Attraction_city_country_region_continentdf= Attraction_city_country_regiondf.merge(continentdf,on='ContinentId',how='left')
    print('Continent....................')
    Attraction_city_country_region_continentdf=Attraction_city_country_region_continentdf.rename(columns={'Continent':'AttractedContinent'})
    print(Attraction_city_country_region_continentdf.head())


    typedf=pd.read_excel("Type.xlsx")
    Attraction_city_typedf=Attraction_city_country_region_continentdf.merge(typedf,on='AttractionTypeId',how='left')
    Attraction_detailsdf=Attraction_city_typedf.drop(columns=["AttractionCityId","AttractionTypeId","CityId","RegionId","ContinentId","CountryId"])
    print("Attraction Details :")
    print(Attraction_detailsdf.head())

    #Main Table 
    print("Transaction Details : ")
    Transactiondf=pd.read_excel('Transaction.xlsx')
    print(Transactiondf.head())

    #Merging Transaction and User details
    Transaction_userdf=Transactiondf.merge(user_detailsdf,on='UserId',how='left')

    #Merging transaction and attraction details 

    Transaction_user_attractiondf=Transaction_userdf.merge(Attraction_detailsdf,on='AttractionId',how='left')

    print(Transaction_user_attractiondf.head())



    modedf=pd.read_excel('Mode.xlsx')
    Transaction_user_attraction_modedf=Transaction_user_attractiondf.merge(modedf,left_on='VisitMode',right_on='VisitModeId',how='left')
    print('Main Dataset')
    print(Transaction_user_attraction_modedf.head())
    Transaction_user_attraction_modedf=Transaction_user_attraction_modedf.drop(columns=["VisitMode_x"])
    Transaction_user_attraction_modedf=Transaction_user_attraction_modedf.rename(columns={"VisitMode_y":"VisitMode"})

    #MainDataset Ready!!!
    Transaction_user_attraction_modedf.to_excel("TourismDataset.xlsx",index=False)


def Datacleaning(tourismdf):
    print(tourismdf.isnull().sum())  #Null Checked 
   
    columns=["VisitYear","VisitMonth","Continent","Region","Country","CityName","Attraction","AttractionAddress","AttractedCity","AttractedCountry","AttractedRegion","AttractedContinent","AttractionType","VisitMode"]
    for col in columns:
        print(col)
        print(tourismdf[col].unique())

    #Outliers Hanndling 
    numerical_columns = ["Rating", "VisitMonth", "VisitModeId"]

    for col in numerical_columns:
        Q1=tourismdf[col].quantile(0.25)
        Q3=tourismdf[col].quantile(0.75)
        IQR=Q3-Q1
        lowerbound=Q1 - (1.5 * IQR)
        upperbound=Q3+(1.5*IQR)
        outliers = tourismdf[(tourismdf[col] < lowerbound) | (tourismdf[col] > upperbound)]
        print(f'Total Outliers{col}: {outliers[col].shape[0]}')
        
    plt.figure(figsize=(10, 6))
    tourismdf[numerical_columns].boxplot()
    plt.title("Box Plot of Numerical Columns")
    plt.show()
    return tourismdf

def DataPreprocessing(tourismdatadf):

#Encoding the categorical variable 
# Initialize separate encoders for each column
    tourismdatadf=tourismdatadf.drop(columns=["AttractedContinent", "AttractedRegion", "AttractedCountry","AttractedCity"])
    # 🔹 Define categorical columns for Target Encoding
    #Feature Engineering
    target_encode_cols = ["Continent", "Country", "CityName", "AttractionType", "Attraction"]
    # 🔹 Apply Target Encoding
    te = TargetEncoder()
    processeddata[target_encode_cols] = te.fit_transform(processeddata[target_encode_cols], processeddata["Rating"])
    
    # 🔹 One-Hot Encode VisitMode
    processeddata = pd.get_dummies(processeddata, columns=['VisitMode'], drop_first=False)
    processeddata=processeddata.drop(columns='VisitMode_Encoded')

    
    tourismdatadf['averageratingpervisitmode']=tourismdatadf.groupby('VisitMode')['Rating'].transform('mean')
    tourismdatadf['averageratingperattraction']=tourismdatadf.groupby('Attraction')['Rating'].transform('mean')
    tourismdatadf.to_excel("ProcessedTourismdata.xlsx",index=False)


def Createsidebar():
    st.sidebar.markdown(f'Hello {os.getlogin()} :)')
    seletedtab=st.sidebar.radio('Go To =>',options=['EDA','Regression Task','Classfier Task','Recommedation system Task','About'])
    return seletedtab

def Filters(df):
    # Date Filters
    st.sidebar.write('Filters : ')
    selected_year = st.sidebar.selectbox("Select Visit Year", df["VisitYear"].unique())
    selected_month = st.sidebar.multiselect("Select Visit Month", df["VisitMonth"].unique())

    # Geographical Filter
    selected_country = st.sidebar.multiselect("Select Country", df["Country"].unique())

    # Attraction Filter
    selected_attraction = st.sidebar.multiselect("Select Attraction", df["Attraction"].unique())

    # User Behavior Filters
    selected_rating = st.sidebar.slider("Select Rating", min_value=1, max_value=5, value=(1, 5))
    selected_visit_mode = st.sidebar.multiselect("Select Visit Mode", df["VisitMode"].unique())
    return selected_year,selected_country,selected_month,selected_rating,selected_attraction,selected_visit_mode

def EDA(tourismdf,processeddf):
    st.header('Exploratory Data Analysis (EDA)')
 
    selected_year,selected_country,selected_month,selected_rating,selected_attraction,selected_visit_mode=Filters(tourismdf)
   
    tourismdf["Rating"] = tourismdf["Rating"].astype(int)

    # Apply filters 
    if selected_year:
        tourismdf = tourismdf[tourismdf["VisitYear"] == selected_year]  

    if selected_month:
        tourismdf = tourismdf[tourismdf["VisitMonth"].isin(selected_month)]

    if selected_country:
        tourismdf = tourismdf[tourismdf["Country"].isin(selected_country)]

    if selected_attraction:
        tourismdf = tourismdf[tourismdf["Attraction"].isin(selected_attraction)]

    if selected_rating:
        tourismdf = tourismdf[(tourismdf['Rating'] >= selected_rating[0]) & 
                            (tourismdf['Rating'] <= selected_rating[1])]

    if selected_visit_mode:
        tourismdf = tourismdf[tourismdf["VisitMode"].isin(selected_visit_mode)]

    st.divider()

    #Metrics
    st.subheader('Tourism Metrics')
    col1,col2,col3=st.columns(3)
    
    highest_rated_place = tourismdf.groupby('Attraction')['Rating'].mean().idxmax()
    highest_rating = tourismdf.groupby('Attraction')['Rating'].mean().max()

    col1.metric("Top Rated Attraction", highest_rated_place, f"⭐ {highest_rating} / 5")

    most_common_mode = tourismdf['VisitMode'].value_counts().idxmax()
    most_common_mode_count = tourismdf['VisitMode'].value_counts().max()

    col2.metric("Most Common Visit Mode", most_common_mode, f"{most_common_mode_count} visitors")

    most_visited_country = tourismdf['Country'].value_counts().idxmax()
    most_visited_country_count = tourismdf['Country'].value_counts().max()

    col3.metric("Most Visited Country", most_visited_country, f"{most_visited_country_count} visits")

    col4, col5 = st.columns(2)

    # Get Most & Least Attracted Places with Visitor Count
    most_attracted = tourismdf.groupby('Attraction')['UserId'].count().sort_values(ascending=False)
    least_attracted = tourismdf.groupby('Attraction')['UserId'].count().sort_values(ascending=True)

    most_attracted_place = most_attracted.idxmax()
    most_attracted_count = most_attracted.max()

    least_attracted_place = least_attracted.idxmin()
    least_attracted_count = least_attracted.min()

    col4.metric('Most Attracted Place', most_attracted_place, f"{most_attracted_count} visitors")
    col5.metric('Least Attracted Place', least_attracted_place, f"{least_attracted_count} visitors")
    st.divider()
   
    # user distribution across continent,countries,regions
    # ---- 1. User Distribution Across Continents ----
    tab1,tab2,tab3=st.tabs(['User Across Continents','User Across Region','User Across Country'])
    with tab1:
        st.subheader("Unique User Distribution Across Continents")

        user_dist = tourismdf.groupby('Continent')['UserId'].nunique().sort_values(ascending=False).reset_index()

        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x='Continent', y='UserId', data=user_dist, ax=ax,palette="viridis")
        ax.set_title("User Distribution Across Continents")
        ax.set_xlabel("Continents")
        ax.set_ylabel("Number of Users")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with tab2:
         
        st.subheader("Unique User Distribution Across Region")


        user_region = tourismdf.groupby('Region')['UserId'].nunique().reset_index()
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x='Region', y='UserId', data=user_region, ax=ax,palette="viridis")
        ax.set_title("User Distribution Across Region")
        ax.set_xlabel("Regions")
        ax.set_ylabel("Number of Users")
        plt.xticks(rotation=60)
        st.pyplot(fig)

    with tab3:
      
        st.subheader("Unique  Top 10 User Distribution Across Country")
        user_country =( tourismdf.groupby('Country')['UserId'].nunique().sort_values(ascending=False).head(10).reset_index())
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x='Country', y='UserId', data=user_country, ax=ax,palette="viridis")
        ax.set_title("User Distribution Across Country")
        ax.set_xlabel("country")
        ax.set_ylabel("Number of Users")
        plt.xticks(rotation=60)
        st.pyplot(fig)


    #Explore attraction types and their popularity based on user ratings
    st.subheader('Attraction Types and their popularity based on user ratings')
    attraction_popularity=tourismdf.groupby('AttractionType').agg(
        Average_Rating=('Rating','mean'),
        Total_visitors=('UserId','count')
    ).sort_values(by="Total_visitors",ascending=False).reset_index()

    st.dataframe(attraction_popularity)

    #VisitMode 
    st.subheader('Correlation Between visitMode and user demographics')
    Visitmodelrelationdf=processeddf.drop(columns=["TransactionId","UserId","VisitYear","VisitMonth","AttractionId","Rating","Attraction","AttractionAddress","AttractionType","VisitModeId",'AverageRatingPerUser','AverageRatingPerVisitMode','AverageRatingPerAttraction','Continent','Region','Country','CityName','Attraction','AttractionAddress','AttractionType','VisitModeId','VisitMode','TransactionId','UserId','AttractionId','Rating','AttractionType_Encoded','Attraction_Encoded','AttractionAddress_Encoded'])
    correlationmatrix=Visitmodelrelationdf.corr()
    fig,ax=plt.subplots(figsize=(8,5))
    sns.heatmap(data=correlationmatrix,annot=True)
    st.pyplot(fig)

    #Alternate Approach to find corelation using chi square distribution
    
    # columns=['Continent','CityName','Country','Region']
    # for col in columns:
    #     contingency_table = pd.crosstab(tourismdf['VisitMode'], tourismdf[col])

    #     # Run Chi-Square Test
    #     chi2, p, dof, expected = chi2_contingency(contingency_table)
    #     print(f" For {col} : Chi-Square Statistic: {chi2}, p-value: {p}")

    st.subheader('Top 15 Ratings based on the Attraction spots.')
    fig, ax = plt.subplots(figsize=(12, 6))
    avaragerating=tourismdf.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(15).reset_index()
    sns.barplot(data=avaragerating, x="Attraction", y="Rating",ax=ax,palette="coolwarm")
    plt.xticks(rotation=60)
    plt.ylabel("Average Rating")
    plt.xlabel("Attraction Spots")
    plt.title("Average Rating by Attraction Spots")

    #Average Ratings across different countries
    st.pyplot(fig)
    st.subheader('Ratings across different Countries.')
    geodf=tourismdf.groupby('Country')['Rating'].mean().sort_values(ascending=False).reset_index()
    fig = px.choropleth(geodf, locations="Country", locationmode="country names", color="Rating",
                        title="Average Rating by Countries", color_continuous_scale="Viridis")
    st.plotly_chart(fig)


def align_features_for_prediction(encoded_input, model, feature_names):
    # Ensure the prediction data has the same VisitMode columns
    for col in feature_names:
        if col.startswith("VisitMode_") and col not in encoded_input.columns:
            encoded_input[col] = 0  # Add missing VisitMode columns with value 0

    encoded_input = encoded_input[feature_names]
    
    return encoded_input

# Load sample data to calculate aggregated features
TourismData = pd.read_excel("ProcessedTourismdata.xlsx")

def get_average_rating(feature, value):
    if value in TourismData[feature].values:
        return TourismData[TourismData[feature] == value]["Rating"].mean()
    return TourismData["Rating"].mean()  # Default mean rating

def GettingInputFromUserAndPredictingRegressionTask():
    # 🔹 Load the saved Model, Encoder & Feature Names
    model = joblib.load("lightgbm_model.pkl")
    te = joblib.load("target_encoder.pkl")
    feature_names = joblib.load("input_features.pkl")
    st.divider()
    # 🔹 Streamlit UI
    st.header("Attraction Rating Prediction 🎡🌍")

    st.write("Enter details to predict the tourist rating!")

    # User Inputs
    continent = st.selectbox("Continent", TourismData["Continent"].unique())
    country = st.selectbox("Country", TourismData["Country"].unique())
    city = st.selectbox("City", TourismData["CityName"].unique())
    attraction_type = st.selectbox("Attraction Type", TourismData["AttractionType"].unique())
    attraction = st.selectbox("Attraction", TourismData["Attraction"].unique())
    visit_mode = st.selectbox("Visit Mode", TourismData["VisitMode"].unique())
    visit_year = st.number_input("Visit Year", min_value=2000, max_value=2030, step=1)
    visit_month = st.number_input("Visit Month", min_value=1, max_value=12, step=1)
    # Compute Aggregated Features
    avg_rating_attraction = get_average_rating("Attraction", attraction)
    avg_rating_visitmode = get_average_rating("Attraction", visit_mode) # Assuming new user

    #  Apply Target Encoding for Categorical Features
    encoded_data = pd.DataFrame([[continent, country, city, attraction_type, attraction]], 
                                columns=["Continent", "Country", "CityName", "AttractionType", "Attraction"])
    encoded_data = te.transform(encoded_data)

    #  One-Hot Encode VisitMode
    visit_mode_df = pd.get_dummies(pd.DataFrame([visit_mode], columns=["VisitMode"]), drop_first=False)

    # Ensure all one-hot encoded VisitMode columns exist
    for col in feature_names:
        if col.startswith("VisitMode_") and col not in visit_mode_df.columns:
            visit_mode_df[col] = 0  # Add missing columns with 0

    # Get categorical feature values after target encoding
    encoded_features = encoded_data.iloc[0, :].values.tolist()

    # Ensure all features are included
    user_input_data = encoded_features + [visit_year, visit_month, avg_rating_attraction, avg_rating_visitmode]
    user_input_data += visit_mode_df.iloc[0, :].values.tolist()  # Add one-hot encoded VisitMode features

    #  Convert into DataFrame with correct feature names
    user_input = pd.DataFrame([user_input_data], columns=feature_names)

    # 🔹 Prediction Button
    if st.button("Predict Attraction Rating"):
        with st.spinner(text='Prediction InProgress'):
            predicted_rating = model.predict(user_input)
            st.success(f"🎯 Predicted Rating: {round(predicted_rating[0], 2)}⭐")

    
def RegressionModel():
    GettingInputFromUserAndPredictingRegressionTask()


#Classifier Task
def GettingInputFromUserAndPredictingClassifierTask():

    ClassfierModelProps=joblib.load('ClassifierModelProperities.pkl')
    loaded_model=ClassfierModelProps['classModel']
    loaded_targetencoder=ClassfierModelProps['classtarget_encoder']
    loaded_labelencoder=ClassfierModelProps['classlabel_encoder']
    loaded_featurenames=ClassfierModelProps['classfeature_names']
    #  Streamlit UI
    st.divider()
    st.header("VisitMode Prediction 🎡🌍")

    st.write("Enter details to predict the Mode Of Visit!")

    # User Inputs
    continent = st.selectbox("Continent", TourismData["Continent"].unique())
    country = st.selectbox("Country", TourismData["Country"].unique())
    city = st.selectbox("City", TourismData["CityName"].unique())
    attraction_type = st.selectbox("Attraction Type", TourismData["AttractionType"].unique())
    attraction = st.selectbox("Attraction", TourismData["Attraction"].unique())
    visit_year = st.number_input("Visit Year", min_value=2000, max_value=2030, step=1)
    visit_month = st.number_input("Visit Month", min_value=1, max_value=12, step=1)
    rating = st.slider("Rating", min_value=1, max_value=5, step=1)

    input_df = pd.DataFrame([{
        "VisitYear": visit_year,
        "VisitMonth": visit_month,
        "Rating": rating,
        "Continent": continent,
        "Country": country,
        "CityName": city,
        "Attraction": attraction,
        "AttractionType": attraction_type
    }])

    # Columns used for target encoding
    categorical_columns = ['Continent', 'Country', 'CityName', 'AttractionType', 'Attraction']
    categorical_input=input_df[categorical_columns]
    numerical_input=input_df.drop(columns=categorical_input)

    #Applying Target Encoding
    encoded_categorical =loaded_targetencoder.transform(categorical_input)

    # Combine with numerical features
    encoded_input = pd.concat([numerical_input.reset_index(drop=True), encoded_categorical.reset_index(drop=True)], axis=1)

    encoded_input=encoded_input[loaded_featurenames]

    if st.button("Predict Mode Of Visit"):
            with st.spinner(text='Prediction InProgress....Please Wait!'):
                predicted_visitmode = loaded_model.predict(encoded_input)
                decoded_visitmode=loaded_labelencoder.inverse_transform(predicted_visitmode)
                st.success(f"🎯 Predicted Visit Mode is : {decoded_visitmode[0]}")

def ClassifierModel():
    GettingInputFromUserAndPredictingClassifierTask()


#  Recommendation System
def recommend_attractions(user_id,user_similaritydf,useritemmatrix,id_to_name,top_n):
    if user_id not in user_similaritydf.index:
        st.warning("No recommendations found. Try a different user.")
        return None
    
    # Get Similar Users (excluding the user itself)
    similar_users = user_similaritydf[user_id].sort_values(ascending=False)[1:]
    # Get attractions rated by similar users
    similar_users_ratings = useritemmatrix.loc[similar_users.index].mean(axis=0)
    # Filter out attractions already rated by the user
    user_rated = useritemmatrix.loc[user_id]
    unrated = user_rated[user_rated.isna()].index
    recommendations = similar_users_ratings.loc[unrated].sort_values(ascending=False).head(top_n)
    # Map to names
    recommendation_list = [ id_to_name.get(attr_id, "Unknown Attraction") for attr_id in recommendations.index]
    # Format Output
    st.subheader(f"🎯 Top {top_n} Recommended Attractions for User ID {user_id}:\n\n")
    sino=0
    for name in recommendation_list:
        sino+=1
        st.write(f"{sino}.{name} \n")

    
def Recommendationsystem(df):
    Fulldatasetdf=df.copy()
    #  Select Relevant Columns
    st.divider()
    st.header('Recommendation System ')
    
    df = df[['UserId', 'AttractionId', 'Rating']]


    df_grouped = df.groupby(['UserId', 'AttractionId'], as_index=False)['Rating'].mean()

    useritemmatrix=df_grouped.pivot(index='UserId',columns='AttractionId',values='Rating')

    sparse_matrix=csr_matrix(useritemmatrix.fillna(0))

    user_similarity=cosine_similarity(sparse_matrix)

    user_similaritydf=pd.DataFrame(user_similarity,index=useritemmatrix.index,columns=useritemmatrix.index)

    # Create mapping for AttractionId -> Attraction name
    id_to_name = Fulldatasetdf.drop_duplicates('AttractionId').set_index('AttractionId')['Attraction'].to_dict()
    col1,col2=st.columns(2)
    with col1:
            userid=st.number_input('UserID',placeholder='Enter the UserID',min_value=1)
    with col2:
        top_Recommendations=st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)
    if st.button('Get Recommendations!'):
        recommend_attractions(userid,user_similaritydf,useritemmatrix,id_to_name,top_Recommendations)


def About():
    
    st.markdown('''
    This application uses data science techniques to understand users better, make predictions, and give smart suggestions. It includes:

    📊 Exploratory Data Analysis (EDA)
    This helps us explore the data using charts and summaries to find patterns, understand relationships, and clean the data before building any models.

    🔍 Regression
    Used to predict rating—like how much rating they may get for this place from (1 To 5) Rating Scale.

    🗂️ Classification
    Used to predict Visit Mode—like whether a person is traveling for business, with family, friends, or as a couple.

    🌟 Recommendation System
    Suggests tourist attractions to users based on their past activity, preferences, and interests.''')



def main():
    st.title('Tourism Analytics 🗺️🔍📈')
    #MainDataset()
    #tourismdatasetdf=pd.read_excel("TourismDataset.xlsx")
    #datadf=Datacleaning(tourismdatasetdf)
    #DataPreprocessing(tourismdatasetdf)
    
    selectedtab=Createsidebar()
    if selectedtab=='EDA':
        tourismdatasetdf=pd.read_excel("TourismDataset.xlsx")
        processeddata=pd.read_excel('ProcessedTourismdata.xlsx')
        EDA(tourismdatasetdf,processeddata)
    elif selectedtab=='Regression Task' : 
         RegressionModel()
    elif selectedtab=='Classfier Task':
        ClassifierModel()
    elif selectedtab=='Recommedation system Task':
        tourismdatasetdf=pd.read_excel("TourismDataset.xlsx")
        Recommendationsystem(tourismdatasetdf)
    else:
        st.header('About')
        st.divider()
        About()




if __name__=="__main__":
     main()