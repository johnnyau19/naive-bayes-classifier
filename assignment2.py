import numpy as np
import pandas as pd

def naive_bayes_classifier(dataset_filepath, snake_measurements=None):
  # Read dataset, set header=None since we dont want pd treats the first line of data as a header
  df = pd.read_csv(dataset_filepath, header=None)

  # Assign name to each column, there is header now
  df.columns = ["class", "length", "weight", "speed"]

  # Disect the dataset into sub-datasets where each contains one class
  grouped = df.groupby("class")

  # Calculate the mean of the length feature of each class
  mean_df = grouped.mean()
  # print(mean_df)

  # Calculate the std of the length feature of each class
  std_df = grouped.std()
  # print(std_df)

  # Prior of each class
  priors = df["class"].value_counts(normalize=True)
  # print(priors)

  # Implement a Gaussian PDF function 
  def gaussian_pdf(x, mean, std):
    return (np.exp((-1/2)*((x-mean)/std)**2))/(np.sqrt(2*np.pi)*std)

  # Likelihood for each class
  likelihoods = {}   #{likelihood_python: P(length|python)*P(speed|python)*P(weight|python), likelihood_anaconda: P(length|anaconda)*P(speed|anaconda)*P(weight|anaconda)}
  features = df.columns[1:]
  classes = df["class"].unique()
  for snake in classes:
    likelihood_of_snake = 1
    for id, feature in enumerate(features):
      x = snake_measurements[id]
      mean = mean_df.loc[snake, feature]
      std = std_df.loc[snake, feature]
      std = 1e-6 if std == 0 else std           # in case the denominator = 0 in the gaussian pdf
      likelihood_of_snake*= gaussian_pdf(x=x, mean=mean, std=std)
    likelihoods[snake] = likelihood_of_snake
  
  # Posteriors
  posterior_of_snakes={}
  for snake in classes:                             # {snake1: posterior of snake1, snake2: posterior of snake2,...}
    posterior_of_snakes[snake] = priors.loc[snake] * likelihoods[snake]   # unormalized posterior

  # Normalize the posteriors
  P_Evidence = sum(posterior_of_snakes.values())
  for snake in classes:
    posterior_of_snakes[snake] = posterior_of_snakes.get(snake)/P_Evidence

  # class_probabilities is a three element list indicating the probability of each class in the order [anaconda probability, cobra probability, python probability]
  class_probabilities = list(posterior_of_snakes.values())
  # most_likely_class is a string indicating the most likely class, either "anaconda", "cobra", or "python"
  most_likely_class = max(posterior_of_snakes, key=posterior_of_snakes.get)   
  
  return most_likely_class, class_probabilities

# print(naive_bayes_classifier("Examples\Examples\Example0\dataset.csv", [350, 42, 13]))