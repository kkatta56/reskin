# Files
## new_snake.py
Run this file in order to move the Dobot in a snake pattern over each of the skins. The snake pattern starts at the bottom left of the skin and makes indentations of a specified depth (while avoiding the screws on the corners) until it reaches the top right corner. It will either return to the bottom left corner to start a new iteration with a different specified indentation depth or move on to the next skin. By default, the Dobot will move 4 millimeters in either the x- or y-direction between indentations, but this value can be changed (see line 52). However, using 2 millimeters between indentations will give more accurate results while taking a reasonable amount of time. Data is collected before every indentation to calculate the baseline level before indentation and also during the indentation.

Look for the section of the code in between two rows of hashes (lines 138-144). The ports refer to the tinker boards, and ‘/dev/ttyACM0’ is the board closest to you when facing the setup. The origin values refer to the bottom left corner of each of the skins.

## process_data.py
Run this file after collecting data with new_snake.py. You will need to change the values in lines 83-85. The time_string variable refers to the name of the folder where the collected data is stored. The num_ports variable is the number of skins/ports in the collected data, and the num_depths variable is the number of depths that were iterated over each skin in the collected data. 

This file will first calculate the average baseline data values from the collected data. It will then subtract the baseline data from the raw data for each indentation and save it in the ‘processed’ folder. It will also normalize the data and store it in the ‘normalized’ folder.

## plot_test.py
This file is self-explanatory. It plots the data for each magnetometer across all of the uploaded skin data sets. It is useful to qualitatively check for any errors in the data collection process.

## model_test.py
This file will train a model and then test it for accuracy depending on various levels of precision. The model used in this file is the same as the one used in the published paper. You can change the model parameters in the model.py file in the utils folder (that file needs to be cleaned up a bit. I didn’t know how to make files interact with each other using import).

The time_string variable in line 146 needs to be changed to the name of the data folder. If you plan on training on certain datasets and testing on another, look at lines 149-153. Add the URLs of the datasets you want to use for training and testing to the train_urls and test_urls variables. If you plan on training and testing on the same dataset, look at lines 158-161. Add the desired datasets to the url variable.

The test function makes predictions with the trained model and compares the values to the actual value. If the predicted value is within a certain distance from the true value, it will be marked as accurate. You can add a list of tolerances (for location and force) in lines 173-174 that the test function will test against. The file will output the results for each combination of location tolerances and force tolerances.

## Data
Location on desktop: code -> kaushik-reskin -> reskin -> datasets
- Base_orientations is just a collection of many tests with the base orientation (to make sure all the data was consistent between trials)
- magnet_switch_1_and_2, etc are datasets collected by switching the skins mentioned
- New_skins_1mm and new_skins_2mm are just base datasets but indentations are made every 1mm and every 2mm, respectively.

## Preliminary Data Analysis
https://docs.google.com/presentation/d/1dBxn1O_xbm9TXJxXlk3bOxSVli7mE59XXX1L30W54mA/edit?usp=sharing

The visualizations show the magnetic flux output value (denoted by y-axis) from each of the 15 magnetometers (denoted by the x-axis) for each skin (denoted by dot color). In an ideal situation, these dots should all be the same value. However, in this situation, there are many anomalies that imply some external factors cause the values to be different for each skin.

However, I had a tough time coming up with useful conclusions from this data analysis, and I seem to have approached this from a direction that may not be fully logical. I recommend trying a different approach to making sure all the skins are behaving similarly.










# ReSkin Project Goals:
- Investigate different elastomers (polyurethanes and silicones) for use with ReSkin
  - Dynamics range
  - Sensitivity
  - Contact resolution
- Characterize ReSkin sensor for performance in external magnetic fields, gripping metal objects, and with multipoint contact
- Asses potential applications for ReSkin
