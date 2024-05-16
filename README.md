# Model Merging 
Model merging is a technique that combines two or more LLMs into one model.

>It’s feasible to merge models with mixing architectures, for example: LLaMA 2 + Mistral + Wizard.

- We can pick the best of both worlds say computer vision and NLP and merge into one. It saves a lot of time instead creating a different architecture for each use case, you only have
to maintain one.
## Merge Algorithm
### 1. EDITING MODELS WITH TASK ARITHMETIC 
- Negating a task vector
  - Forgetting via negation  $\tau_{\text{new}} = -\tau$ corresponds to extrapolating
between the fine-tuned model and the pre-trained model.
- Adding a task vector
  - $\tau_{\text{new}} = \sum_{i} \tau_i$ results in a multi-task model proficient in all tasks, sometimes even
with gains over models fine-tuned on individual tasks.
- Combining a task vector

This study dives deep into the intricate workings of task vectors within pre-trained models, shedding light on their potential to steer model behavior effectively. By constructing task vectors through the contrast of pre- and post-fine-tuning model weights, the research unveils a powerful mechanism for task optimization.

One of the most intriguing aspects is how arithmetic operations like negation and addition can manipulate these task vectors, directly influencing task performance. The revelation that negating a task vector decreases performance while leaving control tasks relatively unaffected highlights the precision with which these vectors can be tuned.

Even more exciting is the discovery that combining task vectors through addition can lead to enhanced performance across multiple tasks simultaneously. This not only underscores the versatility of task vectors but also hints at the possibility of more efficient multitask learning strategies.

Perhaps the most remarkable finding is the application of task vector combinations in tasks linked by analogy relationships. The ability to leverage task vectors from related tasks to improve performance on a fourth task, without any direct training data, opens up a realm of possibilities for transfer learning and generalization.
![](taskarth.PNG)

Firstly, we define **w<sub>pre</sub>** as the weights of a pre-trained model, and **w<sub>ft</sub>** as the corresponding weights after fine-tuning on task *t*. The task vector **v<sub>t</sub>** is essentially the difference between these two sets of weights, calculated element-wise: **v<sub>t</sub>** = **w<sub>ft</sub>** - **w<sub>pre</sub>**.



Now, the interesting part is how these task vectors can be applied to other model parameters $\mathbf{w}$ of the same architecture. This is achieved through element-wise addition, with an optional scaling factor $\lambda$, resulting in a new set of weights $\mathbf{w}_{\text{new}} = \mathbf{w} + \lambda \mathbf{v}$.

### **TIES-MERGING (TRIM, ELECT SIGN & MERGE):** Introduces three novel approaches to solve these problems.

1. **Trim:** For each task t, we trim the redundant parameters from the task vector τt to create ˆτt by
   keeping the top-k% values according to their magnitude and trimming the bottom (100−k)% of
   the redundant parameters by resetting them to 0. This can be decomposed further as ˆτt = ˆγt⊙ˆμt.
2. **Elect:** Next, we create an aggregate elected sign vector γm for the merged model that resolves
   the disagreements in the sign for each parameter p across different models. To create the elected
   sign vector, we choose the sign with the highest total magnitude across all relevant models. For
   each parameter p ∈ {1, 2, . . . , d}, we separate the values {ˆτ p
   t }nt
   =1 based on their sign (+1 or
   −1) and take their sum to calculate the total mass (i.e., total magnitude) in the positive and the
   negative direction. We then assign γpm
   as the sign with greater total movement. This can be
   efficiently computed using γpm
   = sgn(
   Pn
   t=1 ˆτ p
   t ).
3. **Disjoint Merge:** Then, for each parameter p, we compute a disjoint mean by only keeping
   the parameter values from the models whose signs are the same as the aggregated elected sign
   and calculate their mean. Formally, let Ap = {t ∈ [n] | ˆγp
   t = γpm
   }, then τpm
   = 1
   |Ap|
   P
   t∈Ap ˆτ p
   t .    Note that the disjoint mean always ignores the zero values.

>Depiction of steps involved.

![](TIES.PNG)
### INTERFERENCE FROM REDUNDANT PARAMETERS
- However, when merging
a parameter that is influential for one model but redundant
(i.e. not influential) for other models, the
influential value may be obscured by the redundant
values, lowering the overall model performance
- INTERFERENCE FROM SIGN DISAGREEMENT: A given parameter might have a positive
value for some models and a negative value for others. Consequently, employing simple averaging
might compromise the performance on both tasks.
![](conflict.PNG)

- SLERP
- TIES
- DARE
- Passthrough

Define each Algorithm and add paper

![img.png](https://arxiv.org/html/2403.13257v1/extracted/5482855/figures/model_merging_classification.png)
