# Word-limit prompts: first sample per condition

## `wl01` (limit 40 words)

> Explain why regularization can improve generalization in at most 40 words.

**SFT (no DPO)** - 20 words, compliant=True, reward 4.15

> Regularization helps prevent overfitting by penalizing large weights during training, leading to simpler models that generalize better to unseen data.

**Standard DPO** - 22 words, compliant=True, reward 3.90

> Regularization helps prevent overfitting by adding a penalty term to the loss function, encouraging simpler models that generalize better to unseen data.

**Length-balanced DPO** - 17 words, compliant=True, reward 4.43

> Regularization helps prevent overfitting by adding a penalty for complexity, thus improving model performance on unseen data.

## `wl02` (limit 25 words)

> Define overfitting in no more than 25 words.

**SFT (no DPO)** - 23 words, compliant=True, reward 3.90

> Overfitting occurs when a statistical model describes random errors or noise instead of the underlying relationship, resulting in poor performance on new data.

**Standard DPO** - 23 words, compliant=True, reward 3.90

> Overfitting occurs when a statistical model describes random errors or noise instead of the underlying relationship, resulting in poor performance on new data.

**Length-balanced DPO** - 23 words, compliant=True, reward 3.69

> Overfitting occurs when a statistical model describes random errors or noise instead of the underlying relationship, resulting in poor performance on new data.

## `wl03` (limit 30 words)

> Summarize the purpose of a validation set in at most 30 words.

**SFT (no DPO)** - 17 words, compliant=True, reward 3.36

> The validation set is used to tune hyperparameters and evaluate model performance without overfitting on training data.

**Standard DPO** - 17 words, compliant=True, reward 3.38

> The validation set is used to tune hyperparameters and evaluate model performance without overfitting on training data.

**Length-balanced DPO** - 17 words, compliant=True, reward 3.38

> The validation set is used to tune hyperparameters and evaluate model performance without overfitting on training data.

## `wl04` (limit 35 words)

> Explain gradient descent in under 35 words.

**SFT (no DPO)** - 38 words, compliant=False, reward 3.11

> Gradient Descent is an optimization algorithm used to find the minimum of a function by iteratively moving towards the lowest point along the steepest descent direction. It updates weights based on the negative gradient of the cost function.

**Standard DPO** - 51 words, compliant=False, reward 3.50

> Gradient Descent is an optimization algorithm used to find the minimum of a function. It works by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. The update rule adjusts parameters based on this gradient information until convergence or a stopping criterion is met.

**Length-balanced DPO** - 38 words, compliant=False, reward 3.11

> Gradient Descent is an optimization algorithm used to find the minimum of a function by iteratively moving towards the lowest point along the steepest descent direction. It updates weights based on the negative gradient of the cost function.

## `wl05` (limit 45 words)

> Describe one advantage and one limitation of attention in at most 45 words.

**SFT (no DPO)** - 50 words, compliant=False, reward 3.32

> Advantage: Attention mechanisms allow models to focus on relevant parts of the input data during training and inference, enhancing their ability to learn from complex patterns.
> 
> Limitation: Attention can be computationally expensive for large-scale models as it requires significant memory storage and processing power for calculating context-aware weights.

**Standard DPO** - 50 words, compliant=False, reward 3.61

> Advantage: Attention mechanisms allow models to focus on relevant parts of the input data during training and inference, enhancing their ability to learn from complex patterns.
> 
> Limitation: Attention can be computationally expensive for large-scale models as it requires significant memory storage and processing power for calculating context-aware weights.

**Length-balanced DPO** - 50 words, compliant=False, reward 3.40

> Advantage: Attention mechanisms allow models to focus on relevant parts of the input data during training and inference, enhancing their ability to learn from complex patterns.
> 
> Limitation: Attention can be computationally expensive for large-scale models as it requires significant memory storage and processing power for calculating context-aware weights.

## `wl06` (limit 30 words)

> What is distribution shift? Answer in no more than 30 words.

**SFT (no DPO)** - 31 words, compliant=False, reward 3.25

> Distribution shift occurs when the training data and test data have different distributions, causing models trained on one to perform poorly on another due to biased assumptions or incorrect feature representations.

**Standard DPO** - 31 words, compliant=False, reward 3.18

> Distribution shift occurs when the training data and test data have different distributions, causing models trained on one to perform poorly on another due to biased assumptions or incorrect feature representations.

**Length-balanced DPO** - 31 words, compliant=False, reward 3.18

> Distribution shift occurs when the training data and test data have different distributions, causing models trained on one to perform poorly on another due to biased assumptions or incorrect feature representations.

## `wl07` (limit 35 words)

> Explain the role of a learning rate in at most 35 words.

**SFT (no DPO)** - 63 words, compliant=False, reward 4.25

> A learning rate determines how much the model's parameters change with each iteration during training. It controls the step size in gradient descent and helps avoid overshooting the minimum or getting stuck in local minima. A smaller learning rate reduces risk but may slow down convergence; larger rates increase speed but could lead to divergence. Proper selection is crucial for effective training.

**Standard DPO** - 63 words, compliant=False, reward 4.04

> A learning rate determines how much to adjust the model parameters for each update during training. It controls the step size in gradient descent and helps avoid overshooting the minimum or getting stuck in local minima. A smaller learning rate reduces risk but may slow down convergence; larger rates increase speed but can lead to divergence. Proper selection is crucial for effective training.

**Length-balanced DPO** - 63 words, compliant=False, reward 4.08

> A learning rate determines how much to adjust the model parameters for each update during training. It controls the step size in gradient descent and helps avoid overshooting the minimum or getting stuck in local minima. A smaller learning rate reduces risk but may slow down convergence; larger rates increase speed but can lead to divergence. Proper selection is crucial for effective training.

## `wl08` (limit 40 words)

> Why can a model with high training accuracy still fail at deployment? Answer in at most 40 words.

**SFT (no DPO)** - 46 words, compliant=False, reward 3.20

> A model may have high training accuracy due to overfitting during training, which means it performs well on the training data but poorly on new, unseen data (deployment). This is why deploying such models without proper validation or adjustments for generalization issues can lead to failures.

**Standard DPO** - 46 words, compliant=False, reward 3.49

> A model may have high training accuracy due to overfitting during training, which means it performs well on the training data but poorly on new, unseen data (deployment). This is why deploying such models without proper validation or adjustments for generalization issues can lead to failures.

**Length-balanced DPO** - 46 words, compliant=False, reward 3.53

> A model may have high training accuracy due to overfitting during training, which means it performs well on the training data but poorly on new, unseen data (deployment). This is why deploying such models without proper validation or adjustments for generalization issues can lead to failures.

## `wl09` (limit 30 words)

> Define calibration for a probabilistic classifier in under 30 words.

**SFT (no DPO)** - 23 words, compliant=True, reward 3.90

> Calibration involves adjusting the confidence levels of a probabilistic model's predictions to match their actual probabilities, ensuring better decision-making and reliability.

**Standard DPO** - 22 words, compliant=True, reward 4.21

> Calibration involves adjusting the confidence scores of predictions to reflect their actual probability estimates, ensuring that predicted probabilities closely match true likelihoods.

**Length-balanced DPO** - 22 words, compliant=True, reward 4.16

> Calibration involves adjusting the confidence scores of predictions to reflect their actual probability estimates, ensuring that predicted probabilities closely match true likelihoods.

## `wl10` (limit 45 words)

> Explain what a frozen reference model does in RLHF in at most 45 words.

**SFT (no DPO)** - 64 words, compliant=False, reward 2.19

> In Reinforcement Learning with Human Feedback (RLHF), a frozen reference model is one that remains unchanged throughout the training process. This means it doesn't update based on interactions with the environment or feedback from humans. Instead, it serves as a baseline or starting point for learning, helping to stabilize and improve models through supervised learning techniques such as fine-tuning human-generated examples.

**Standard DPO** - 69 words, compliant=False, reward 2.75

> In Reinforcement Learning with Human Feedback (RLHF), a frozen reference model is one that remains unchanged throughout the training process. This means it uses the same policy and reward function as the original model during both human feedback collection and fine-tuning phases. The goal is to learn from the expert demonstrations while preserving the learned behavior of the original model, making the transition between the two more seamless.

**Length-balanced DPO** - 63 words, compliant=False, reward 3.01

> In Reinforcement Learning with Human Feedback (RLHF), a frozen reference model is one that remains unchanged throughout the process of fine-tuning for a specific task or policy. This approach aims to leverage existing knowledge and performance by not modifying it during training, potentially leading to faster convergence and better performance compared to models that can be updated dynamically based on human feedback.

