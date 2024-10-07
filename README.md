# IISER Mohali Poker Bot

Welcome to the IISER Mohali Poker Bot repository! This project focuses on building an advanced poker-playing AI capable of autonomous decision-making with superhuman performance. This poker bot project is made by **Team Code of Cards** for the competition **Bet and Byte** at IISER Mohali's fest **Insomnia 2024**. Below, you'll find an overview of the project's structure, the research behind our approach, and how our implementation operates. **This is the first open-source implementation of the ReBeL algorithm for a poker bot.**

## Repository Structure

This repository contains the following files:

1. **4.ipynb** - This notebook contains the model training code. We experimented with various reinforcement learning techniques, including Counterfactual Regret Minimization (CFR) and its advanced variants, to train our poker bot effectively.
2. **5.ipynb** - Here, the model is executed and tested in a live poker environment. This notebook contains the primary gameplay logic and utilizes the trained model for decision-making in different poker scenarios.
3. **modules.py** - A modularized Python file containing important functions from 4.ipynb that are necessary for running 5.ipynb. By storing key functions separately, we facilitate cleaner imports and reusable code across different notebooks.
4. **policy_net.pth** - This file contains the pre-trained weights for the policy network used in our poker bot. The policy network is responsible for guiding the bot's decision-making process by determining the best moves based on the game state.
5. **value_net.pth** - This file contains the pre-trained weights for the value network. The value network helps the bot assess the potential outcome of its decisions, optimizing its strategy over time.

**This is the first open-source implementation of the ReBeL algorithm for a poker bot**, making this project a unique contribution to the open-source community.


## Research Background

We built our bot upon foundational concepts from game theory and state-of-the-art algorithms tailored for imperfect-information games like poker. Here’s a summary of the methodologies and algorithms that informed our approach:

### Game Theory and Poker AI

1. **Counterfactual Regret Minimization (CFR)** - We started by understanding CFR, a popular algorithm for finding Nash equilibria in sequential games. This algorithm iteratively minimizes regret to make optimal decisions over time, which is particularly valuable in poker, where moves are made based on probabilistic beliefs about opponents' cards.
   
2. **Monte Carlo CFR (MCCFR)** - As an extension, we used MCCFR, which enhances CFR by utilizing Monte Carlo sampling, allowing our bot to handle more complex decision trees efficiently.

3. **Deep CFR** - For a more optimal approach, we employed Deep CFR, which integrates neural networks with CFR. This version of the algorithm enables our bot to learn continuously from large data sets, improving performance as it encounters more gameplay scenarios.

### ReBeL Model

ReBeL (Recursive Belief-based Learning) is an advanced model developed by Facebook AI Research to tackle imperfect-information games like poker. It combines deep reinforcement learning with search strategies to approximate Nash equilibria, delivering superhuman performance. **This is the first open-source implementation of the ReBeL algorithm for a poker bot.**  The model leverages **public belief states** (PBS) to track the common knowledge about the game's progress, allowing our bot to make more informed decisions.

Key features of the ReBeL model include:

- **Self-Play and Search:** ReBeL trains itself by playing against various policies, refining its strategy using a policy network for efficient searches during gameplay.
- **Imperfect-Information Handling:** Unlike previous models (like AlphaZero), ReBeL adapts to environments where not all information is available, as is the case in poker.
- **Superhuman Performance:** Through rigorous self-play and deep learning, ReBeL achieves a level of play that can outperform human experts in poker, even with limited domain-specific knowledge.

For more on ReBeL, refer to the detailed research [here](https://arxiv.org/pdf/2007.13544).

## How It Works

Our poker bot makes real-time decisions during gameplay by calculating expected values and updating strategies based on counterfactual regret. Here’s a breakdown of the main functionalities:

1. **Hand Evaluation:** Using 5.ipynb, the bot evaluates each hand based on probabilities derived from past experiences and self-play data.
   
2. **Betting Strategy:** The bot adjusts its betting strategy dynamically, responding to opponents' moves and making use of the PBS framework for probabilistic decision-making.

3. **Improvement Over Time:** Through repeated simulations, the bot continually enhances its strategy, reducing errors and refining its ability to anticipate opponents' moves.

## Future Enhancements

While our current bot performs at a high level, there are several potential improvements we may pursue:
- **Multi-player Adaptation:** Extending the model to handle multi-player poker games, as the current version is focused on heads-up scenarios.
- **Incorporation of Advanced Search Techniques:** Implementing more advanced search algorithms during training could further optimize the bot's strategy.
- **Broader Game Applications:** Adapting ReBeL to work with other imperfect-information games beyond poker, such as strategy-based board games.

## Conclusion

Our poker bot represents a significant step forward in the application of AI to complex decision-making environments. By combining CFR algorithms with cutting-edge deep reinforcement learning techniques, we have created a bot that not only understands the fundamentals of poker but also adapts and improves over time.

We hope this project contributes to the broader field of AI research in game theory and imperfect-information games. Thank you for reviewing our work, and feel free to reach out for any questions or collaborations!
