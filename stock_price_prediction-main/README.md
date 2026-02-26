# stock_price_prediction
Developed a machine learning-powered web application leveraging Long Short-Term Memory (LSTM) models to accurately predict future stock prices. The app combines advanced time-series analysis with a user-friendly interface, enabling insightful decision-making for traders and investors. 

# INTRODUCTION

Stock price prediction is a crucial aspect of financial markets, providing valuable insights for investors and traders. Traditional methods often fail to capture the complex and volatile nature of stock prices. This project leverages Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), known for its ability to learn long-term dependencies in sequential data. By utilizing historical stock data from Yahoo Finance, the LSTM model forecasts future stock prices with enhanced accuracy. Additionally, a web application developed using Streamlit offers an interactive platform for users to input stock IDs and visualize predictions. This project aims to combine advanced machine learning techniques with practical application, offering a robust tool for stock market analysis.

## 1.1.	PROBLEM STATEMENT
Accurately predicting stock prices is a significant challenge due to the inherent volatility and complexity of financial markets. Traditional statistical methods often fail to capture the dynamic and nonlinear patterns in stock price movements. This project seeks to address this issue by developing a robust predictive model using Long Short-Term Memory (LSTM) networks. Additionally, there is a need for an accessible platform that provides real-time stock price predictions. By incorporating moving averages and deploying the model in a user-friendly web application, this project aims to enhance the accuracy of stock price predictions and provide valuable insights for investors and traders.


## 1.2.TECHNIQUES INVOLVED
This project employs a combination of data science and machine learning techniques to predict stock prices and deploy the model for real-time use. The key techniques involved are:
1.	Data Collection:
o	Yahoo Finance API: Historical stock data is sourced from Yahoo Finance using the yfinance Python library. This includes data such as opening price, closing price, highest and lowest prices, trading volume, and adjusted closing prices over a specified period.
2.	Data Preprocessing:
o	Handling Missing Values: Missing data points are identified and handled to ensure the dataset is complete and consistent.
o	Feature Engineering: Additional features like moving averages (e.g., 100-day and 250-day moving averages) are calculated and included to help the model identify trends.
o	Normalization: Stock prices are normalized using MinMaxScaler to scale the data between 0 and 1, which helps in improving the model's convergence during training.
3.	Model Building:
o	Long Short-Term Memory (LSTM) Networks: An LSTM network, a type of Recurrent Neural Network (RNN), is used due to its ability to capture long-term dependencies and patterns in sequential data. The model is constructed using the TensorFlow and Keras libraries.
o	Model Architecture: The LSTM model comprises multiple layers, including LSTM layers with various units, followed by Dense layers to produce the final output.
4.	Model Training:
o	Train-Test Split: The dataset is split into training and testing sets to evaluate the model's performance on unseen data.
o	Batch Training: The model is trained in batches over several epochs to optimize the learning process and improve accuracy.
o	Loss Function and Optimizer: The model uses Mean Squared Error (MSE) as the loss function and the Adam optimizer for efficient gradient descent.
5.	Model Evaluation:
o	Performance Metrics: The model's performance is evaluated using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) to assess prediction accuracy.
o	Inverse Transform: Predicted values are inverse transformed to their original scale to compare against actual stock prices.
6.	Visualization:
o	Matplotlib: The matplotlib library is used to create visualizations of the stock prices, moving averages, and predicted values, providing a clear view of the model's performance.
7.	Deployment:
o	Streamlit: The trained model is deployed using Streamlit, a Python library for creating interactive web applications. The web app allows users to input stock IDs and view real-time predictions and visualizations of stock price trends.


# LITERATURE REVIEW

## 1. Introduction to Stock Price Prediction
Stock price prediction remains a complex and high-stakes challenge due to the volatility and unpredictability of financial markets. Traditionally, statistical methods such as ARIMA and GARCH have been employed for forecasting. However, advancements in machine learning and deep learning have introduced more sophisticated techniques with the potential to enhance prediction accuracy.
## 2. Historical Stock Price Data
Historical stock price data is foundational for predictive modeling. It includes features such as closing prices, trading volumes, and moving averages. The author suggested that stock prices follow a random walk, indicating that past prices do not predict future prices. Despite this, models leveraging historical data can reveal patterns and trends that inform predictions. (Fama 1970)
## 3. Data Preprocessing
Data preprocessing, including normalization and scaling, is essential for preparing data for machine learning models. In the provided code, Min-Max scaling is used to normalize stock price data, ensuring equal contribution of all features during model training.(Mele 2012)
## 4. Moving Average Techniques
Moving averages, such as Simple Moving Averages (SMA) and Exponential Moving Averages (EMA), are widely used in technical analysis to smooth price data and identify trends. Malkiel (2003) explores how moving averages help in recognizing long-term trends and making trading decisions. The code utilizes 100-day and 250-day moving averages to capture both short-term and long-term trends in stock prices.
## 5. Machine Learning Models
Machine learning models, particularly deep learning approaches like Long Short-Term Memory (LSTM) networks, have gained prominence for time series forecasting. LSTMs, introduced by “Hochreiter and Schmidhuber (1997)”, address the vanishing gradient problem in traditional RNNs, making them effective for sequential data such as stock prices. The provided code applies an LSTM model to predict future stock prices based on historical data.
## 6. Case Studies
### 1.	Case Study 1: Stock Price Prediction Using LSTM Networks
The use of LSTM networks for stock price prediction. The authors compared LSTMs with other machine learning models and found that LSTMs significantly outperformed traditional models in terms of prediction accuracy. This case study supports the choice of LSTM networks in the provided code for predicting stock prices. (Fischer and krauss 2018)
### 2.	Case Study 2: Web-Based Stock Prediction Tools
The study highlighted the advantages of deploying machine learning models through interactive web applications, allowing users to input stock symbols and view real-time predictions. This case study underscores the value of the Streamlit-based web app included in the provided code.(Liaw et al 2015)
### 7. Evaluation Metrics
Evaluating the performance of predictive models is crucial for assessing their accuracy. Mean Squared Error (MSE) is a common metric used to measure the difference between predicted and actual values. MSE as a reliable measure for forecasting models. The code uses MSE to evaluate the accuracy of the LSTM model's predictions.( Chai and Draxler 2014)
### 8. Web Application for Stock Price Prediction
Deploying predictive models via web applications facilitates user interaction and real-time predictions. the integration of machine learning models into web-based tools. The code includes a Streamlit application that allows users to input stock symbols and view predictions, demonstrating practical implementation of these models.


# PROPOSED METHODOLOGY


1. Environment Setup
•	Install Required Libraries:
o	Use pip to install necessary libraries: yfinance, sklearn, scikit-learn, matplotlib, keras, and tensorflow.
2. Data Collection
•	Fetch Historical Stock Data:
o	Use the yfinance library to download historical stock price data. For instance, download data for Google (ticker symbol: "GOOG") from 20 years ago to the present date.
3. Data Exploration and Visualization
•	Examine Data:
o	Review the first few rows, shape, statistical summary, and information of the dataset to understand its structure and content.
•	Check Missing Values:
o	Identify any missing values in the dataset.
•	Plot Time Series Data:
o	Plot the adjusted closing price over time to visualize stock price trends.
4. Feature Engineering
•	Calculate Moving Averages:
o	Compute moving averages for 100 and 250 days to understand longer-term trends.
•	Calculate Percentage Change:
o	Compute daily percentage changes in the adjusted closing price.
•	Normalize Data:
o	Normalize the adjusted closing prices using MinMaxScaler from sklearn to scale values between 0 and 1.
5. Prepare Data for Modeling
•	Create Sequences:
o	Prepare sequences of 100 days of historical prices as input features (x_data) and the next day's price as the target variable (y_data).
•	Split Data:
o	Split the data into training and testing sets (70% training and 30% testing).
6. Model Building
•	Define LSTM Model:
o	Construct a Sequential LSTM model using Keras:
	Two LSTM layers with 128 and 64 units respectively.
	Two Dense layers for output prediction.
•	Compile Model:
o	Compile the model using the Adam optimizer and Mean Squared Error loss function.
•	Train Model:
o	Train the model on the training data for a specified number of epochs.
7. Model Evaluation
•	Make Predictions:
o	Use the trained model to predict stock prices on the testing data.
•	Inverse Transform Predictions:
o	Convert the normalized predictions and actual values back to the original scale using the scaler.
•	Calculate RMSE:
o	Compute the Root Mean Squared Error (RMSE) to evaluate model performance.
•	Visualize Results:
o	Plot the original test data and predictions to visualize the model’s performance.
8. Model Deployment
•	Streamlit Web Application:
o	Create a web app using Streamlit to deploy the model.
o	Allow users to input stock ticker symbols.
o	Fetch and display historical stock data using yfinance.
o	Load the saved model and make predictions on the data.
o	Plot the results using matplotlib to visualize predictions and historical data.
9. Save and Load Model
•	Save Model:
o	Save the trained model to a file using Keras’s model.save() method.
•	Load Model:
o	Load the saved model for making predictions in the web application.


# IMPLEMENTATION

## FRONTEND DEVELOPMENT:
1. User Interface Design
•	Title and Instructions: Create a title for the web app (e.g., "Stock Price Predictor App") and provide clear instructions for users to input the stock symbol.
•	Input Field: Implement an input field where users can enter the stock symbol they want to predict (e.g., "Enter the Stock ID").
2. Data Display
•	Stock Data: Display the historical stock data fetched from Yahoo Finance in a tabular format so users can review the data used for predictions.
3. Visualization
•	Prediction vs. Actual Plot: Generate and display a plot comparing the predicted stock prices with the actual prices. Use different colors or styles to distinguish between predicted and actual values.
•	Interactive Plotting: Ensure the plot is interactive and updated based on the stock symbol entered by the user.

## BACKEND DEVELOPMENT:
1. Data Retrieval
•	Fetch Stock Data: Use the Yahoo Finance API (yfinance) to download historical stock data based on the user-provided stock symbol. Set the date range to cover the last 20 years.
2. Data Preprocessing
•	Scaling: Apply Min-Max scaling to normalize the stock prices, preparing the data for the model.
•	Feature Engineering: Transform the scaled data into sequences suitable for LSTM model input.
3. Model Handling
•	Load Model: Load the pre-trained LSTM model from disk to make predictions on the stock data.
•	Prediction: Use the model to predict future stock prices based on the processed input data.
4. Post-Processing
•	Inverse Scaling: Convert the scaled predictions back to their original price scale for accurate comparison.
•	Result Preparation: Create a data frame with actual and predicted stock prices for visualization and download purposes.
5. Integration and Deployment
•	Streamlit Integration: Integrate all components (data retrieval, preprocessing, prediction, and visualization) into a Streamlit app to provide a seamless user experience.
•	Model Management: Ensure the model is correctly saved and loaded, and handle any potential issues related to model compatibility or data format.



# RESULTS AND DISCUSSIONS

## Data Overview:
•	Data Shape and Summary: The historical stock data for Google (GOOG) was successfully retrieved for the past 20 years. The dataset includes features such as 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume'.
## Data Visualization:
•	Stock Price Trends: The adjusted closing prices over time were plotted, showing the historical trends and fluctuations of Google's stock price.
•	Moving Averages: Two moving averages (100-day and 250-day) were plotted alongside the adjusted closing prices. These moving averages help smooth out short-term fluctuations and highlight longer-term trends.
## Predictions and Performance:
•	Predictions: The model generated predictions for the test set, which were then inverse-transformed to original scale prices.
•	Comparison with Actual Prices: A plot comparing predicted and actual stock prices was created, illustrating the model’s performance in forecasting.
## Model Evaluation:
•	Root Mean Squared Error (RMSE): The RMSE was calculated to measure the difference between the predicted and actual prices. A lower RMSE indicates better model performance, but exact values should be reviewed to assess accuracy.
## Results Visualization:
•	Prediction vs. Actual Plot: The plot showed how well the model's predictions matched the actual stock prices. This visual representation helps in understanding the accuracy of the model's forecasts.
## Model Performance:
•	Accuracy: The accuracy of the LSTM model is visually assessed through the comparison plot. The closer the predicted prices are to the actual prices, the better the model's performance.
•	Training Time and Epochs: The model was trained for a relatively short period (2 epochs). For improved performance, further tuning of the training duration, batch size, and epochs might be required.
## Data and Feature Engineering:
•	Data Scaling: Min-Max scaling was used to normalize the stock prices. This approach ensures that the input data is within a range suitable for the LSTM model.
•	Feature Engineering: Using moving averages as features could improve model performance by incorporating trend information. Further exploration of additional features might enhance prediction accuracy.

## Model Complexity:
•	LSTM Architecture: The chosen architecture with two LSTM layers and two Dense layers is relatively simple. Experimenting with more complex architectures or hyperparameters might yield better results.
•	Overfitting and Generalization: It is crucial to ensure that the model generalizes well to unseen data. Regularization techniques and more extensive validation could help in achieving this.
## Practical Application:
•	Utility of Predictions: The model provides a basic framework for stock price prediction, which can be useful for traders and analysts. However, stock price prediction is inherently complex and influenced by numerous factors beyond historical prices.
•	Further Improvements: Incorporating additional data such as trading volumes, news sentiment, and macroeconomic indicators could improve the model's forecasting ability.
## User Interaction and Experience:
•	Streamlit Interface: The Streamlit app provides a user-friendly interface for interacting with the model. Users can input stock symbols, view predictions, and download results, making it accessible for non-technical users.
## Future Improvements
1.	Model Enhancements: Experiment with more complex architectures and hyperparameter tuning to boost accuracy.
2.	Feature Expansion: Include additional features like trading volume and external data for better predictions.
3.	Data Quality: Use more comprehensive datasets and handle missing values more effectively.
4.	User Experience: Improve visualizations and incorporate real-time data for a more interactive experience.

<img width="915" height="647" alt="image" src="https://github.com/user-attachments/assets/f3d77f47-43bf-4a4f-a806-499630091119" />
<img width="915" height="954" alt="image" src="https://github.com/user-attachments/assets/0e4e4934-0145-4f68-9247-4bdd958fee5a" />
<img width="915" height="912" alt="image" src="https://github.com/user-attachments/assets/69edd62b-e5d5-4b8c-b62e-35e8126ea4d6" />
<img width="915" height="1062" alt="image" src="https://github.com/user-attachments/assets/1452c533-5c2f-4956-944f-b798c83c3951" />
<img width="915" height="919" alt="image" src="https://github.com/user-attachments/assets/3531409e-a248-408e-ba6c-fe11ecd3e804" />




# Conclusion
The current stock price prediction model provides a solid foundation for forecasting, using historical data with LSTM. Future improvements can enhance accuracy and user experience, making the model more robust and valuable for real-world applications.

OUTPUT
 
 
 
 
 
