Identification and Classification of Gambling Dice using Image Feature Detection and a Convolutional Neural Network

By Jesse Sheehan <jps111@uclive.ac.nz>
Student ID: 53366509

Report:
	The report can be found as a PDF file in the report.pdf file.
	I feel sorry for whoever has to mark this, it made a really good 3 page article but lost some of its charm at 6 pages. Sorry about this.

Demonstration:
	This is found inside "Demonstration.mkv".
	It is a 30 second video demonstrating how the program operates.
	The video was created with OBS Studio.

Code:
	The code can be found inside the "code" directory.

	The main program can be run by executing "python main.py".
	How to drive this program:
	- "q" for quit.
	- "p" for pause/unpause. I used this for taking screenshots for the report and for collecting data for the results section.
	- "s" for save. This will extract each segmented die face and save it in its own file inside the "die_images" folder. These are used by the classification program.

	Other command line utilities have also been provided:
	- "train.py" for training the model.
	- "classify.py" for manually classifying data that has been saved with "main.py".
	- "resize.py" for automatically resizing and grayscaling classified images (this shouldn't be needed anymore as "classify.py" now does this job)

	Some explanations for the other python files:
	- "Gui.py" is a small framework for generating GUIs with trackbars so I could manually tune parameters.
	- "data.py" contains functions for loading and saving CNN model data.
	- "utilities.py" contains an assortment of helper functions.

	- The "camera_matrix.npy" and "distortion_coeff.npy" files contain information on removign distortion from the Logitech C170 webcam.

	The "model.ckpt" directory contains the trained model for the CNN.
	The "training_data" directory contains the pre-classified image data for the "train.py" program.
