# Overview

### Many old men are bald
- Psychological Induction  
&nbsp; - inductive statement based on experience  
&nbsp; - also has certain predictive aspect    
&nbsp; - no scientific explanation  

- Statistical View  
&nbsp; - the lack of hair = random variable  
&nbsp; - estimate its distribution (depending on age) from past observations (traning sample)  

- Philosophy of Science Approach  
&nbsp; - find scientific theory to explain the lack of hair  
&nbsp; - explanation itself is not sufficient  
&nbsp; - true theory needs to make non-trivial predictions  

### Conceptiual Issues
* Any theory (or model) has two aspects:
1. explanation of past data (observations)
2. prediction of future (unobserved) data

* For a model to acheive both goals (explaination and prediction) perfectly is not possible
* Important issues to be adressed:
&nbsp; - quality of explanation and prediction  
&nbsp; - is good prediction possible at all?  
&nbsp; - if two models explain past data equally well, which one is better?  
&nbsp; - how to measure model complexity  

### Induction:
Induction is the process of inferring a general law or principle from the observations of particular instances

### Ockham's Razor
A problem solving principle which suggests that we should reduce assumptions to their minimum.

### Expected outcomes
#### Scientific / Technical
- Learning = generalization, concepts and issues
- Math theory: Statistical Learning Theory aka VC-Theory
- Conceptual basis vor various learning algorithms

#### Methodological
- How to use available statistical/machine learning/ data mining 
- How to compare prediction accuracy of different learning algorithms
- Are you getting good modeling results because you are smart or just lucky

#### Practical Applicaitons:
- Financial engineering
- Biomedical + Life Sciences
- Security
- Image recognition etc.

### Promise of Big Data
- More Data -> More knowledge

### Scientific Discovery
- Combines ideas/models and facts/data
#### First principle knowledge:
&nbsp; hypothesis -> experiment -> theory  
&nbsp; deterministic, casual, intelligible models  

#### Modern data-drien discovery:
&nbsp; s/w program + DATA -> knowledge  
&nbsp; statistical, complex systems  

- Many methodological differences

### Invariants of Scientific Knowledge
- Intelligent questions
- Non-trivial predictions
- Clear limitations / constraints
- All require human intelligence
&nbsp; - missing / lost in Big Data  

### Common Goals of Modeling
- Prediction (Generalization)  
- Interpretation - descriptiv model
- Human decision-making using bot above
- Information retrieval, i.e. predictive or descriptiv modeling of unspecified subset of available data

### Three distinct Methodologies 
#### Statistical Estimation
- from classical statistics and fct approximation
#### Predictive Learning (~machine learning)
- practitioners in machine learning / neural networks
- Vapnik-Chervonenkis (VC) theory for estimating predictive models from empirical (finite) data samples
#### Data Mining
- exploratory data analysis, i.e. selecting a subset of available (large) dataset with interesting properties

### General Experimental Procedure
1. Statement of the Problem
2. Hypothesis Formulation (Problem Formalization) -  different from classical statistics
3. Data Generation / Experiment Design
4. Data Collection and Preprocessing
5. Model Estimation (learning)
6. Model Interpretation, Model Assessment ad Drawing Conclusions

Note: 
&nbsp; - each step is complex and requires several iterations  
&nbsp; - estimated model depends on all previous steps  
&nbsp; - observational data (not experimental design)  

### Data Preprocessing and Scaling
- Preprocessing is required with observational data 
Examples:
- Basic preprocessing includes
&nbsp; - summary univariate statistics: mean, st. deviation, min+max value, range, boxplot performed independently for each input/output  
&nbsp; - detection (removal) of outliers  
&nbsp; - scaling of input/output variables (may be necessary for some learning algorithms)  
- Visual inspection of data is tedious but useful

### Cultural + Ethical Aspects
- Cultural and business aspects usually affect:
&nbsp; - problem formalization  
&nbsp; - data access / sharing (i.e., in life sciences)  
&nbsp; - model interpretation  

- Possible (idealistic) solution approach
&nbsp; - to adopt common methodology  
&nbsp; - critical for interdisciplinary projects  
