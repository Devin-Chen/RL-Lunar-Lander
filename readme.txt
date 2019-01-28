# rldm-project-2
Project 2 source code for CS7642, Summer 2017

Github Source Code: https://github.gatech.edu/zchen482/rldm-project-2

Video Presentation: https://youtu.be/x8SppvHqq80

#### author: Zhi Chen(ID:zchen482)
#### date: 7/1/2017

### Environment
requires: python 3.6; numpy 1.13; tensorflow 1.2; keras 2.0.5; latest OpenAI gym 

### Result Output
Training and testing data results are stored in .txt in folder "./src/results/[timestamp]/“.
OpenAI gym monitor results are stored in .txt in folder "./src/monitor/[timestamp]/“.
DQN network weights are saved in .txt in folder "./src/monitor/[timestamp]/“.

### Example Result
The example result folder located at “./src/example_result/“ consists of a result instance for one of experiment performed to solve Lunar Lander.

### Instructions
1. To reproduce experiment 1 and 2, go to main.py and uncomment generate_exp1_n_2() method call. Then, Run main.py. This generates training and testing data in .txt format stored at "./src/results/[timestamp]“. DQN weights are stored at "./src/monitor/[timestamp]/“. Monitor results are stored at "./src/monitor/[timestamp]/“.

2. To reproduce experiment 3, go to main.py and uncomment generate_exp3() or generate_exp3_full() method call. Then, Run main.py. This generates training and testing data in .txt format stored at "./src/results/[timestamp]“ for each a single experiment with specified parameters or the full experiment 3 detailed in report. DQN weights are stored at "./src/monitor/[timestamp]/“. Monitor results are stored at "./src/monitor/[timestamp]/“.
