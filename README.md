## Introduction 🔖

The goal of this challenge is to build a Machine Learning model to predict the genres of a movie, given its synopsis. Your solution will be evaluated on the performance of your Machine Learning model and on the quality of your code.

To succeed, you must implement a Python package called `challenge`, which exposes an API with two endpoints. Calling the first endpoint (`/genres/train`) should create and train your model on the data provided. Calling the second endpoint (`/genres/predict`) should use the former model to create predictions on the provided synopses and return these results.

On a high-level, the challenge is divided into three different tasks, each with a corresponding estimated effort:

1. Read the documentation (15min)
2. Set up the environment (15min)
3. Create and train your model (4h)
4. Set up the `/genres/predict` endpoint to create predictions using your trained model (1h)
5. Improve code quality (30min)

## Creating a solution 🏗

### Implementation

To solve this challenge, you are going to implement a Python package called `challenge` that exposes an API. This API must be implemented using [FastAPI](https://fastapi.tiangolo.com/), and should expose two different endpoints:

1. A training endpoint at `localhost:9876/genres/train` to which you can POST a CSV with header `movie_id,synopsis,genres`, where `genres` is a space-separated list of movie genres. This data should be used to create and train your model.
2. A prediction endpoint at `localhost:9876/genres/predict` to which you can POST a CSV with header `movie_id,synopsis`. Your model, trained in the previous endpoint, should use this data to predict movie genres ranked by their likelihood. Once finished, this endpoint should return a dictionary in the following format: `{<movie-id>: {0: <first-prediction>, 1: <second-prediction>, 2: <third-prediction>, 3: <fourth-prediction>, 4: <fifth-prediction>}, ...}`.

The data used to train your solution will be downloaded automatically when running `./run.sh`.

<details>
<summary>Here you can find an extensive list of the tasks you need to implement in order to complete the challenge:</summary>

0. Run `init.sh` to create a virtual environment in which the code can run
1. In the `/genres/train` endpoint:
   1. Create a model
   2. Train the model on the received data
   3. Save the model
2. In the `/genres/predict` endpoint:
   1. Create the endpoint
   2. Load in the previously trained model
   3. Make predictions (ranked) on the received data
   4. Return your predictions in dictionary-format, as specified above
3. Run `run.sh` to evaluate your implementation
</details>

### Files 

You should implement your code in the `challenge/` folder, where it is not allowed to add any files outside this folder. When doing so, you are free to add new files but please don't remove any. If you want to use dependencies that are not yet supported by this package, you can add these to `environment.yml`. However, please don't remove any of the pre-existing dependencies since this might break the code.

It is not allowed to change the bash files (`init.sh` and `run.sh`) and the `setup.cfg` file, since these are used to evaluate your solution. The last section addresses these files in more detail, however, it is not required for you to understand these scripts in order to solve the challenge.


## Running and scoring of your solution 💯

<details>
<summary>Every time you run `./run.sh`, your solution will run and gets evaluated.</summary>

1. Download the `train.csv` and `test.csv` datasets
2. Start your FastAPI server on port 9876
3. POST `train.csv` to `localhost:9876/genres/train` to train your model
4. POST `test.csv` to `localhost:9876/genres/predict` to create a `submission.json` with the top 5 most probable genres (ranked) for each test movie
5. Stop your FastAPI server once complete, or when either training or evaluation fails
6. Compute a score that indicates the quality of your code
7. Upload `submission.json` to our evaluation endpoint to get a score on your predictions
8. Geometrically combine both of your scores: code quality score (6) and predictive score (7)
9. Ask for your git username and email address, if not yet configured
10. Print your final score and send the results to us for validation
</details>

<details>
<summary>Your solution will be evaluated on the performance of your Machine Learning model and on the quality of your code. Once you achieve the target score (see Introduction), one of our engineers will review your code. Based on that review, we will set up an interview with you. We will evaluate your final commit to your repository, so please make sure it runs properly.</summary>

The final score is the geometric mean of two components:
1. Your **predictive score** evaluated as the [Mean Average Precision at K](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)) for the top 5 predicted genres. Note that this metric keeps the ranking into account.
2. Your **code quality score**, which is the geometric mean of:
   1. Whether you added files outside the `./challenge` folder: `0%` if you did, `100%` otherwise
   2. A percentage score based on `flake8`
   3. A percentage score based on `isort`
   4. A percentage score based on `pydocstyle`
   5. A percentage score based on `mypy`
   6. A percentage score based on the actual number of lines of code
   
