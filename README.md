# Starcraft II Machine Learning Bot

AI arena Bot page : [sc2-raiden-bot](https://aiarena.net/bots/601/)

Welcome to the repository for my Starcraft II Machine Learning (ML) bot, developed for competing in a Starcraft II bot ladder competition. This bot is designed using Python and leverages various libraries and frameworks to train and deploy an AI agent capable of playing Starcraft II autonomously.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Setup Instructions](#setup-instructions)
4. [Bot Training](#bot-training)
5. [Deployment](#deployment)
6. [Important Considerations](#important-considerations)
7. [Acknowledgments](#acknowledgments)

## Introduction

This project is inspired by the Sentdex tutorial on integrating Machine Learning with the Starcraft II API. However, due to the rapidly evolving nature of ML frameworks and tools, much of the code and techniques used in the tutorial have become outdated. This repository provides an updated implementation with additional insights gained during the development and training process.

## Project Overview

The bot is built using the [BurnySc2 python-sc2 framework](https://github.com/BurnySc2/python-sc2), which provides an interface for creating and training Starcraft II bots in Python. The bot's development was a challenging yet rewarding process, involving extensive use of tools like ChatGPT to assist in solving complex mathematical problems, optimizing pathfinding algorithms, and cleaning up the code.

### Key Features:
- **SC2 API Integration**: Utilizes the python-sc2 framework to interact with the Starcraft II game.
- **Machine Learning**: The bot is trained using ML techniques to improve its decision-making and gameplay over time.
- **Docker Integration**: Docker was used to manage the environment and facilitate the bot's training.
- **Multi-bot Training**: The bot was trained using a Docker Compose setup running multiple instances simultaneously, allowing for faster and more efficient training.

## Setup Instructions

### Prerequisites
- **Python 3.8+**
- **Starcraft II installed**
- **Starcraft II Bots Maps [Maps wiki](https://aiarena.net/wiki/maps/)**
- **Docker & Docker Compose**

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Olivier-aka-raiden/sc2-raiden-bot.git
   cd sc2-raiden-bot
   ```

2. **Install Python dependencies**: TODO
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Docker**:
   - Build and run the Docker containers as specified in the [BurnySC2 dockerfiles](https://github.com/BurnySc2/python-sc2/tree/develop/dockerfiles).
   - Visit this project to get a docker image of a example environment allowing you to make bots play against eachother : [local-play-bootstrap](https://github.com/aiarena/local-play-bootstrap?tab=readme-ov-file#local-play-bootstrap)
   - For training, use the provided `docker-compose.yml` to launch multiple bots for concurrent training sessions.

5. **Run the Bot**:
   ```bash
   python main_model.py
   ```

## Bot Training

Training the bot involved understanding and applying several ML concepts not fully covered in the Sentdex tutorial. These include:
- **Epochs**: Iterations over the entire dataset during training.
- **Test Data & Shuffling**: Ensuring the bot's training data is shuffled and divided appropriately between training and testing.
- **Numpy Arrays & Tensor Dimensions**: Managing data in the correct format for training models.
- **Learning Rate & Evolutionary Algorithms**: Fine-tuning the bot's learning parameters for optimal performance.

### Training Setup
To train the bot:
1. **Prepare the Training Environment**:
   - Use the BurnySC2 Docker container for a consistent environment.
   - Train multiple bots simultaneously using Docker Compose.

2. **Monitor Training**:
   - Only wins are counted as successful training outcomes.
   - After 24 hours of training, the bot achieved nearly 400 wins, but the training process was not flawless. High variance in results highlighted areas for improvement.

## Deployment

After training, the bot was tested against other bots using the [AIArena local server](https://github.com/aiarena/local-play-bootstrap). To register your bot for competitions, ensure it functions correctly in this local environment before submitting it to the ladder.

## Important Considerations

To achieve successful bot training and deployment, keep the following in mind:
- **Distinct Decisions**: Ensure each decision the bot can make is distinct to avoid confusion during gameplay.
- **1D Array Decisions**: Stick to a single-dimension array for decisions; 2D arrays complicate the model and reduce performance.
- **Bug-Free Scripts**: Minimize bugs to ensure reliable bot behavior.
- **Heatmap Accuracy**: Use heatmaps effectively to guide the bot's decisions.
- **Linux Environment**: Prefer using Google Colab or Linux environments for ML tasks to avoid Windows-related issues.

## Acknowledgments

This project was heavily inspired by the work of Sentdex and the developers of the python-sc2 framework. Special thanks to ChatGPT for assisting in code development and optimization.
