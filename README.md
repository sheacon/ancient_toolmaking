# Ancient Toolmaking Detection Project

- This project is a binary classification problem, using particle analyzer data from archaeological samples to distinguish between ordinary soil particles and stone microdebitage particles associated with ancient toolmaking sites.
- After optimizing and comparing Logistic, Naive Bayes, SVMs, and tree-based models, XGBoost performs the best on the validation set.
- Tuning efforts focused on increasing recall while maintaining precision. 

<br>
<br>

![eda](readme_images/eda.png)
![final_performance](readme_images/.png)

# Background

- The analysis of lithic microdebitage can illuminate 7th century ancient stone tool manufacturing practices to provide insight into past cultural activity. The purpose of this project is to identify the location of ancient stone tool manufacturing areas within the Mayan site of Nacimiento in the Petexbatun region of Guatemala. 
- The locations of these manufacturing areas may be uncovered by analyzing the soil composition. Although ancient stoneknappers cleared large debris from their work area, lithic microdebitage (< 4mm) would be very difficult to remove. Soil samples from 50 locations within the village were collected for analysis.
- A particle analyzer was used to provide 40 measurements of each particle.

# Contributions
- This is an extensively re-worked and extended personal fork of an academic group project. Credit to Mubarak Ganiyu, Sydney Simmons, Shuyang Lin, and Weixi Chen for initial work.
