# SocioEconomic-Status-Prediction-using-Household-Images

Precise and comprehensive measurements of socioeconomic status are crucial for both academic investigations and policy-making. However, in developing countries such measures are available at the local household level only at extremely low frequencies for eg: via a decadal census. A number of papers have attempted predicting economic status at aggregated geographical levels such as district or neighborhood using Deep Learning on images with varying degrees of success. However the utility of such an approach at the household level remains open. In this study we utilize Deep Learning models on household images collected from four northeastern states in India to assess the feasibilty and ethics of household level income status prediction. We categorize households into classes based on income and then train a Swin Transformer model with cross-entropy loss and triplet loss to predict the socioeconomic class of the household. We then compare the prediction accuracy of our model with predictions using a simple list of household assets and predictions from a set of expert human annotators. We find that the use of Deep Learning on images does not lead to any substantial gains in prediction accuracy. Further, we note that human accuracy on this prediction tasks is low, raising questions on the information contained within the images. Our study raises important questions regarding the ethical implications of utilizing household images for predicting socioeconomic status. We explore these ethical implications, emphasizing the importance of a cautious and considerate approach in incorporating image-based techniques.

Paper Link: https://dl.acm.org/doi/10.1145/3675160

Cite: 
@article{dar2024assessing,
  title={Assessing the Feasibility and Ethics of Economic Status Prediction using Deep Learning on Household Images},
  author={Dar, Aatif Nisar and Sengupta, Nandana and Arora, Chetan},
  journal={ACM Journal on Computing and Sustainable Societies},
  year={2024},
  publisher={ACM New York, NY}
}
