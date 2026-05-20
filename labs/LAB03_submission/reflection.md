

| Genre | Part A Accuracy | Part B Accuracy |
|---|---|---|
| Animation | 89.3% | 83.3% |
| Comedy | 80.7% | 74.0% |
| Documentary | 83.3% | 86.7% |
| Horror | 73.3% | 80.0% |
| Romance | 59.3% | 62.0% |
| Sci-Fi | 75.3% | 69.3% |
| **Overall** | **76.9%** | **75.9%** |

Then address these questions:

1. **Architecture choices**: Describe the image branch and tabular branch architectures you settled on. Why did you choose this structure? What did you try that didn't work as well?

2. **Overfitting**: Did you observe a gap between training and validation accuracy? At what point did it appear? What strategies did you use to combat it (dropout, weight decay, early stopping, smaller vocabulary, reduced model size, learning rate scheduling)? Which were most effective?

3. **Part A vs. Part B**: How did your custom CNN compare to the pretrained ResNet18? Did transfer learning help, and if so, in what way (higher accuracy, faster convergence, less overfitting)?

4. **Tabular branch insights**: Which metadata features seemed most useful for genre prediction? Look at the per-class accuracy table — which genres did the model struggle with most? Does that make sense given the available features? If you tried ablations (tabular-only or image-only), what did you learn?

5. **What would you do differently?** If you had more compute time or training data, what would you try next?

6. *(Optional — only if you completed optional extensions)* **Optional extensions**: For each optional experiment you ran, briefly describe what you tried, what result you got, and how it compared to your Part A baseline.

Reflection:

For the image branch, I landed on 4 conv blocks after running 28 short experiments using `experiment_runner.py`. 4 blocks came out 0.4pp ahead of 3, and skip connections, which I expected to help, dropped accuracy by 1.4pp, so I cut them. The tabular branch keeps all four list fields, plus mpaa_rating, each with its own embedding table, and max-pools across tokens, followed by a two-branch MLP that processes numeric features and embeddings separately before merging. Flat concatenation of the two directly lost 2.8pp relative to the split design, and mean pooling lost 1.6pp relative to max, both confirmed in the ablations. The biggest surprise was weight decay: cutting it from 1e-3 to 1e-4 was the single largest gain across all 28 experiments (+0.9pp). Looking at the training logs, validation accuracy consistently outperformed train accuracy in the early epochs, indicating the model was underfitting rather than overfitting. The train-val gap only opened around epoch 16 and peaked at 4.5pp when early stopping fired at epoch 29. Both dropout reduction and a higher LR (1e-3 vs. 3e-4) helped for the same reason.

Part A finished at 76.9% test accuracy; Part B (frozen ResNet18) hit 75.9%. Transfer learning didn't win, but the speed gap was real. Part B's best validation accuracy came at epoch 8, compared to epoch 21 for Part A, with half the trainable parameters. The per-class numbers are more telling than the overall. ResNet18 was better on Horror (+6.7pp) and Documentary (+3.4pp), both genres with photographic, real-world poster imagery that maps well to ImageNet features. It fell behind in Animation (−6pp), Comedy (−6.7pp), and Sci-Fi (−6pp), where poster aesthetics lean toward graphic design and illustration, styles that the backbone was never trained in. The frozen weights that help with photography actively hurt with stylized art.

Romance was the hardest genre in both parts (59.3% Part A, 62.0% Part B), which tracks. Romance posters look a lot like comedy or drama posters, and the metadata doesn't reliably separate them either. Every field subset tested hurt accuracy, so all four list fields stayed in the final model. Given more compute, the first thing I would try is unfreezing ResNet18's layer 4 with a lower LR. The frozen backbone was clearly mismatched for animated poster styles, and partial fine-tuning might close that gap without catastrophic forgetting. I would also rerun the cosine annealing experiment at full training length. At 6 epochs, it looked bad, but that was probably just the schedule decaying LR before the model had time to learn anything.z