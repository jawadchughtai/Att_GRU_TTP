
# An attention-based GRU for TTP

This work contains the implementation of an attention-based gated-recurrent unit approach for short-term travel-time prediction. 
In this work, we have predicted short-term travel time for then next 15 minutes, 30 minutes, 45 minutes and 1 hour based on 1 hour historical data. 
The data is normalized to [0,1] interval. The proposed approach is compared with historical average, support vector regression, eXtreme gradient boosting, multi-layer perceptron, and a gated recurrent unit approach. 
The results are evaluated using root-mean squared error, mean absolute error, mean absolute percentage error, and coefficient of determination (r2). 
The results demonstrate the superior performance of proposed att-GRU as compared to baseline implementations.

## Documentation
The dataset used in this code can be obtained from the link given below:

[Dataset](https://drive.google.com/file/d/1AN7nTsqZe3oPwZr4b5kEEq78HwiMCjMO/view?usp=sharing): This dataset contains 15,073 road segments covering approximately 738.91 km. They are all in the 6th ring road (bounded by the lon/lat box of <116.10, 39.69, 116.71, 40.18>), which is the most crowded area of Beijing. The traffic speed of each road segment is recorded per minute. To make the traffic speed predictable, for each road segment, we use simple moving average with a 15-minute time window to smooth the traffic speed sub-dataset and sample the traffic speed per 15 minutes. Thus, there are totally 5856 ($61 \times 24 \times 4$) time steps, and each record is represented as road_segment_id, time_stamp ([0, 5856)) and traffic_speed (km/h).



## Code
The structure of code:

 - [main.py](https://github.com/jawadchughtai/Att_GRU_TTP/blob/main/main.py): contains implementaiton of GRU and att_GRU.
 - [RNN.py](https://github.com/jawadchughtai/Att_GRU_TTP/blob/main/RNN.py): contains the definitions of GRU implementaion.
 - [plotting.py](https://github.com/jawadchughtai/Att_GRU_TTP/blob/main/plotting.py): contains the details about the graphs generated in our implementation.
 - [baseline_implementations.py](https://github.com/jawadchughtai/Att_GRU_TTP/blob/main/baseline_implementations.py): contains the implementation of baseline appraoches including HA, ARIMA, SVR, XGB, and MLP.


## Deployment

To deploy this project first create the environment using the below command and execute main.py.

```bash
  conda create -n <environment-name> --file Requirements.txt
```


## Screenshots

![15-minute prediction (overall)](https://github.com/jawadchughtai/Att_GRU_TTP/blob/main/Results/att_GRU/att_GRU_qt_lr0.001_bs32_GRUunit32_len_hist4_len_pred1_numepoch600/test_all15.jpg)
![15-minute prediction (two-days)](https://github.com/jawadchughtai/Att_GRU_TTP/blob/main/Results/att_GRU/att_GRU_qt_lr0.001_bs32_GRUunit32_len_hist4_len_pred1_numepoch600/test_twodays15.jpg)

