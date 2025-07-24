# Double-Agent-RL-TicTacToe
This project tries to explore how we can make strategist players by training RL model against eaach other.
The main approach taken here is that the training is done in several iteration. In the first iteration both models are simultaneously trained against each other, after that models are alternatively trained: One of the model is fixed and its Q-Table is not updated while other model learns against it in the next iteration these roles are reversed and it continues untill all iterations are over.

The idea behind this approach is that after models get familiar with basic play, it should be trained against a stable model so that it constantly learns new skills instead continuously changing to fit to dynamic model. This way in each iteration the learning model learns new skills and in succeeeding iteration the next model must learn how to tackle it and learn new moves itself. This way their performance would be generalized.

# Possible Future Steps:
1)Test these models against models not trained on iteration approach.
2)Test this against human players.
3)Test out of one step training model or iterated training model which performs better against humans.
4)Implement this approach on more suitable environments with more potential to learn strategies.
