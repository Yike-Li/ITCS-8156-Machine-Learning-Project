# ITCS-8156-Machine-Learning-Project
In this project, we explored using sine functions to model the yearly seasonal effect of load demand. We used a regional load series and weather data from a US regulated utility and evaluate the model’s performance based on the ex-post and ex-ante forecast. Depending on data availability, we established a formal comparison between the models using calendar month (Month models) or sine functions (VE models) to model the yearly seasonality of load. The recency effect was added to incorporate past temperature information and improve forecasting accuracy. Our experiment results show that VE leads to superior forecasting accuracy over the alternative Month models. The proposed VE models are also easy to interpret and simple and transparent to implement. Regarding the model size, VE involves fewer parameters to estimate than the corresponding Month models and thus saves computation time.

Below are some more elaboration on the files in the folder:
- EDA.py is the EDA analysis code.
- Main_MLR.py gives the main body of the code.
- Model_Fit_Func.py stores all the self-defined methods. 
- Project input.xlsx stores the dates of Equinoxes and Solstices. 
- wss_list.txt stores the weather station selection result. 
