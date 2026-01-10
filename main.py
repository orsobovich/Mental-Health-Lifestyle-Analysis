


#exploration


#data_cleaning.py


#Checks correlation       
logging.info("correlation between Stress Level and Sleep Hours:")        
visualize_correlation(df['Stress Level'], df['Sleep Hours'])

logging.info("correlation between Age and Sleep Hours:")
visualize_correlation(df["Age"], df["Sleep Hours"]) 

logging.info("correlation between Social Interaction Score and Stress Level:")
visualize_correlation(df["Social Interaction Score"], df["Stress Level"])

logging.info("correlation between Age and Social Interaction Score:")
visualize_correlation(df["Age"], df["Social Interaction Score"])

#anova


#visualization.py

