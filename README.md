# Restaurant rating predictor - candidate task by Robert Peetsalu

##DATA INSPECTION
First I checked the relevant features: They originate from only 3 tables geoplaces2, chefmozparking and rating_final and are joinable by placeID. As they have no missing values I assume that future data will also have little missing values, if any. So imputation won't be necessary and the model will just skip rows with missing values in the future. 

Some unmentioned business rules can be extrapolated and checked for in the data, like that each client can only give one rating to each restaurant.

Restaurants with more parking types and more client feedback will have a bigger weight in predicting future ratings, because they'd have more rows per restaurant in the training set, but that's OK since no single restaurant dominates ratings noticeably and there aren't that many parking types to begin with. Judging by the current dataset only a fraction of restaurants have  2 or 3 parking types selected instead of 1. Conveniently none of them have any client feedback, so parking type won't actually skew restaurant representation in the current dataset. 

What could induce bias though is skewed representation of client ratings (254 zeros, 421 ones and 486 twos). Better represented ratings get a bigger classification confidence and thus win in classification more frequently. If the current dataset is a good representation of future rating proportions, then there's no need to correct, as we will want to classify more frequent values more confidently. As there's no way for me to know if the dataset represents future proportions well, I'll suppose that's the case. 

##PREPROCESSING
Here I reused some of my previous projects for speed and quality. Luckily the data structure is very simple and clean, so no problems here.

##SERVING RESTAURANT CLASSIFICATION THROUGH AN API
As this was the aspect of the task that contained unknown unknowns for me (never needed to write API-s before), I wrote it before any ML code. It took some time to learn Flask and API basics, but got done eventually. Requests to localhost port 5000 containing required restaurant features as a list in a JSON string will get a response with predicted ratings. Multiple restaurant ratings can be requested simultaneously. Other requests will get a bad_request_error response in JSON with hints to what's wrong.

##ALGORITHM SELECTION AND PARAMETRIZATION
As performance is said not to matter, any classification algorithm will do. I went with LinearSVC. It does multiclass classification by dividing the feature space with 3 hyperplanes - one between each two restaurant rating categories.

##PIPELINE
Each run of main.py retrains the model with current data. Features are pre-selected and don't need normalization nor standardization, so there's not much to do to improve pipeline accuracy and I skipped using Pipeline class altogether. 
