# A Timeline for "Active Learning for Crowd-Counting"

* Target Conference: AAAI 2024 (Deadline: August 15, 2023)

* Overall idea:
  
  * Proposing an Active learning framework for crowd-counting based on
    * Training time confusion to identify the most difficult samples
    * Finding samples close to the most difficult samples
    * (Optional) Wasserstein distance to identify the most uncertain samples

* Papers to compare with 
  
  * [Active Crowd Counting with Limited Supervision](https://arxiv.org/pdf/2007.06334.pdf) [Primary]
  * [Crowd Counting with Decomposed Uncertainty](https://arxiv.org/abs/1903.07427)
  * [Uncertainty Estimation and Sample Selection for Crowd Counting](https://arxiv.org/abs/2009.14411)

* Things to do:
  
  * [x] Read the papers
  * [ ] <mark>Implement the baseline (Random sample selection and match what is reported in AC-AL paper) </mark>
  * [ ] Analyze the predictions on the training samples across all epochs.
  * [ ] Look at the loss trajectory and find the right epoch to start calculating the confusion.
  * [ ] Implement the confusion calculation
  * [ ] Implement the Wasserstein distance calculation
  * [ ] Implement the sample selection

* Datasets to run experiments on:
  
  * [ ] <mark>ShanghaiTech Part A</mark>
  * [ ] ShanghaiTech Part B
  * [ ] UCF_CC_50
  * [ ] DCC
  * [ ] Mall
  * [ ] TRANCOS
  * [ ] IDCIA

* Tentative plan
  
  | Date        | Task                                                        |
  | ----------- | ----------------------------------------------------------- |
  | June 6-16   | Start implementing the baseline and Look at the predictions |
  | June 17-23  | Implement the confusion calculation                         |
  | June 24-30  | Implement the Wasserstein distance calculation              |
  | July 1-3    | Run experiments on ShanghaiTech Part A                      |
  | July 4      | Start writing the paper                                     |
  | July 4-7    | Run experiments on ShanghaiTech Part B                      |
  | July 8-11   | Run experiments on UCF_CC_50                                |
  | July 12-15  | Run experiments on DCC                                      |
  | July 16-19  | Run experiments on Mall                                     |
  | July 20-23  | Run experiments on TRANCOS                                  |
  | July 24-27  | Run experiments on IDCIA                                    |
  | July 31     | First draft complete                                        |
  | August 1-7  | Revise the paper                                            |
  | August 8    | Abstract due                                                |
  | August 9-14 | Revise the paper                                            |
  | August 15   | Submit the paper                                            |
  | August 18   | Supplementary material and code due                         |
