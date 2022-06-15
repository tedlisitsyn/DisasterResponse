# Disaster Response Pipeline Project

When it is critical situation it is important to understand the meaning of the emergecny message as fast as possible - preferably in the version of web tool so it can't be easily accessable from any place - and act based on this classification.
"Disasters" is the tool where an emergency worker can input a new message and get classification results in several categories.

### Tool Usage:

Use Input Bar to Classify Message:
![Input Message](https://github.com/tedlisitsyn/DisasterResponse/blob/main/screenshots/disasterinput.png)

Get actual information on Distribution of Message Genres:
![Input Message](https://github.com/tedlisitsyn/DisasterResponse/blob/main/screenshots/distmesgenres.png)

Get actual information on Top-10 Responses Categories:
![Top1-0 Responses](https://github.com/tedlisitsyn/DisasterResponse/blob/main/screenshots/top10resp.png)

### Tool Installation:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. From Udacity workspace
	- Run your app with python run.py command (from app folder)
	- Open another terminal and type env|grep WORK this will give you the spaceid (it will start with view*** and some characters after that)
	- Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id that you got in the step 2
	- Press enter and the app should now run for you

2. From Local Machine
	- Once your app is running (python run.py from app folder)
	- Go to http://localhost:3001 and the app will now run

### Files Structure:
app
- template
	- master.html # main page of web app
	- go.html # classification result page of web app
- run.py # Flask file that runs app

data
- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py # get raw data, clean it, and save to the ready for processing DataBase
- DisasterResponse.db # database to save clean data to

models
- train_classifier.py # train the model based on clean data
- classifier.pkl # saved model, current file is to large to upload to GitHub, so can be found [here](https://drive.google.com/file/d/1R9L3nnsuEFkoYPJzXEtQM34uclFBVE5b/view?usp=sharing)

README.md

### Credits
This software uses the following open source packages:
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SQLalchemy](https://www.sqlalchemy.org/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [json](https://www.json.org/json-en.html)
- [plotly](https://plotly.com/)
