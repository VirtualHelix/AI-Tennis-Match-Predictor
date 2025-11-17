# AI-Tennis-Match-Predictor
This project develops a predictive system for ATP tennis match outcomes using all available match data from 1995 to the present. It combines machine learning techniques with custom data structures, including binary search trees and balanced trees, to efficiently store and retrieve large historical datasets. The goal is to create a scalable and accurate model that produces pre-match win probabilities for any given player matchup.

# Overview
Professional tennis produces decades of structured match data containing player statistics, ranking information, surfaces, tournament levels, and contextual features. This project leverages those historical records to train algorithms capable of estimating the probability that Player A defeats Player B under specific conditions. In addition to standard machine-learning workflows, the project incorporates data-structure-based optimizations. Binary trees, AVL trees, and hybrid hash-tree lookup systems allow fast, logarithmic-time retrieval of player histories, surface statistics, head-to-head records, and Elo-ratings snapshots. These structures significantly improve feature computation and reduce the runtime of repeated model queries.

# Key Features
- Historical Dataset (1995–Present)
  - Ingests and processes ATP match results across all tournament levels.
  - Includes match metadata such as date, surface, round, ranking, scorelines, and tournament category.
- Data Processing and Cleaning
  - Standardizes player names across decades.
  - Removes or flags walkovers, retirements, and incomplete matches.
  - Handles missing values and inconsistent records.
- Feature Engineering
  - Player form metrics (overall and surface-specific).
  - Rolling win percentages.
  - Ranking and age differences.
  - Head-to-head statistics.
  - Surface-adjusted Elo ratings.
  - Tournament difficulty indicators.
  - Contextual match features (best-of-3 vs best-of-5, round, tournament prestige).

# Data Structures

The project includes several custom implementations to improve data retrieval:
- Binary Search Trees (BSTs):
  Used for storing chronologically ordered player performance metrics, enabling rapid lookup of recent matches and form calculations.
- AVL / Balanced Trees: 
Used for maintaining sorted match histories across large datasets, ensuring O(log n) updates and retrieval.
- Hybrid Hash-Tree Player Index:
Combines hashing for quick name resolution with a tree-structured history for ordered statistics.

# Elo Rating Tree
A structure that maintains player Elo states and updates them after each match using a logarithmic access pattern.
- These systems support fast computation during:
   - Feature generation
   - Training runs
	 - Real-time prediction calls

# Machine Learning Models
- The modeling pipeline supports multiple algorithms:
  - Logistic regression baseline
  - Random Forest classifier
  - Gradient Boosting (XGBoost, LightGBM, or CatBoost)
  - Custom tree-based predictor that integrates your binary-tree data structures
  - Optional neural network extensions

# All models are trained chronologically to avoid future-data leakage (e.g., train 1995–2018, validate 2019–2021, test 2022–present).
- Evaluation Tools
  - Accuracy, log loss, and AUC.
  - Time-based cross-validation.
  - Calibration curves to evaluate probability reliability.
  - Comparison with baseline predictors (rank and Elo).
