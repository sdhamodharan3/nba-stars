**Team Members: Srinath Dhamodharan, Aiza Aslam, Shelby Crisp, Nealie Glasser**

![Infographic](NEW_NBA.JPG)

# Introduction
## Background

Basketball has become one of the most popular sports worldwide. While there are leagues across the world, many atheletes train to play in the NBA, where the most elite players compete for a chance at a championship. The success of each team is typically seen to rest on the performance of each team's star players, taking into account their abilities to score, create plays, and defend.

## The Problem
While some players are seen better than others and some teams are considered favorites in their games, the outcomes of games are never certain. There are numerous characteristics that determine a team and player's performance and the winner of each game.

## Importance
A model that can accurately predict the outcome of a game based on the stars can provide a lot of insight on the head-to-head between the stars. This can have huge impacts on the game itself knowing what might be optimal strategies for a certain team's offensive strategy and the opposing team's plan to counter these stars. For each franchise it could impact roster decisions to move certain players to improve the team's odds at winning. In addition, successful predictions can have an impact on the business side for bets.

## Goal
We will be isolating the top two to three players on each team and analyzing their most relevant statistics over the past few NBA seasons to predict the outcome of NBA games.

# Methods (Expected)
## Unsupervised
For our unsupervised methods, we expect to use K-Means and PCA for dimensionality reduction and feature correlation. We expect that these methods will be able to cluster our data to identify the most relevant features of a player's stats that affect the outcome of the game. This way, we can reduce the dimensionality of our problem by eliminating certain features for our supervised learning that we find don't significantly impact who wins each NBA game.

## Supervised
For our supervised methods, we plan on exploring several methods such as Neural Networks, random forests, and SVM. We will use our data to train these methods and comparing the results to see if our model can succesfully predict the outcome of a game based on the top players' performance.

# Results
The results we hope to achieve would be an accurate prediction of which team wins based on the performance from their top two or three players. We will measure our success by comparing our results with the Vegas odds for the game as well as also comparing against the actual outcome of these games. 

# Discussion
We hope that our model and methods become an accurate predictor for NBA game outcomes based on just the star players' performances on either team. While we don't expect it to be a perfect predictor, we hope that it's accurate enough. This would allow teams to decide on various teammaking and playmaking strategies around these all-stars and would also impact the players' values and betting. Depending on the results, there may be ways to improve our model/methods and possibly expand this application to other team based sports.

# References
Bunker, Rory P., and Fadi Thabtah. “A Machine Learning Framework for Sport Result Prediction.” Applied Computing and Informatics, vol. 15, no. 1, 19 Sept. 2017, pp. 27–33., doi:10.1016/j.aci.2017.09.005. 

Fayad, Alexander. “Building My First Machine Learning Model: NBA Prediction Algorithm.” Medium, Towards Data Science, 12 July 2020, towardsdatascience.com/building-my-first-machine-learning-model-nba-prediction-algorithm-dee5c5bc4cc1. 

Goitia, Francisco. “An Attempt to Predict the NBA with a Machine Learning System Written in Python Part II.” Medium, HackerNoon.com, 29 Apr. 2019, medium.com/hackernoon/how-to-predict-the-nba-with-a-machine-learning-system-written-in-python-part-ii-f276b19520b9. 
