# DQN-for-job-shop-scheduling
This project uses Deep Q-Network(DQN) for job shop scheduling in Reinforcement learning, and the information is taken from the public data platform - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt.

# Background and Motivation
In industrial manufacturing, production scheduling is a task of significant importance. As mentioned in referenced articles, contemporary demands on such tasks, like job shop scheduling, require systems capable of accommodating various variables in production needs. These variables include production requirements, processing times, and available machines. Obtaining efficient scheduling results in a short time frame for large-scale job shop scheduling presents a considerable challenge.

In the realm of scheduling solutions, universal algorithms such as Simulated Annealing (SA) and Genetic Algorithms (GA) are commonly employed today. Indeed, these methods can provide excellent solutions, but they require redesign when the structure of the problem changes. Their generality is limited, and their performance varies greatly with different problem objectives. On the other hand, deep learning utilizes neural networks for decision-making and can indeed address issues faced by the former methods. However, due to the limitations in the size of datasets, it is more suitable for small-scale problems and not ideal for practical scheduling challenges.

Reinforcement Learning (RL), as a learning algorithm that considers long-term objectives, is a mathematical model that contemplates the relationship between states and actions and is capable of learning. Compared to the previous two methods, it can be applied to scheduling in various scenarios and respond quickly. According to this reference paper, it achieves high-speed, effective scheduling adaptable to various situations. In this project, we implemented DQN, GA, and LPT for sequential scheduling comparison, proving that DQN is superior to the other two methods.

# Methodology

# Data Collection and Analysis Result

# Reference
- Luo, S., Zhang, L. X., & Fan, Y. S. (2021). Dynamic multi-objective scheduling for flexible job shop by deep reinforcement learning. Computers & Industrial Engineering, 159, Article 107489. https://doi.org/10.1016/j.cie.2021.107489 
- Yang, H. B., Li, W. C., & Wang, B. (2021). Joint optimization of preventive maintenance and production scheduling for multi-state production systems based on reinforcement learning. Reliability Engineering & System Safety, 214, Article 107713. https://doi.org/10.1016/j.ress.2021.107713



