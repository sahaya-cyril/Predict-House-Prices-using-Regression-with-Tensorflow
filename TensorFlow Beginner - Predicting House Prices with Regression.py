{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task 1: Introduction\n",
    "\n",
    "---\n",
    "\n",
    "For this project, we are going to work on evaluating price of houses given the following features:\n",
    "\n",
    "1. Year of sale of the house\n",
    "2. The age of the house at the time of sale\n",
    "3. Distance from city center\n",
    "4. Number of stores in the locality\n",
    "5. The latitude\n",
    "6. The longitude\n",
    "\n",
    "![Regression](images/regression.png)\n",
    "\n",
    "Note: This notebook uses `python 3` and these packages: `tensorflow`, `pandas`, `matplotlib`, `scikit-learn`."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.1: Importing Libraries & Helper Functions\n",
    "\n",
    "First of all, we will need to import some libraries and helper functions. This includes TensorFlow and some utility functions that I've written to save time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Libraries imported.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "\n",
    "from utils import *\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout\n",
    "from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback\n",
    "\n",
    "%matplotlib inline\n",
    "tf.logging.set_verbosity(tf.logging.ERROR)\n",
    "\n",
    "print('Libraries imported.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task 2: Importing the Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.1: Importing the Data\n",
    "\n",
    "The dataset is saved in a `data.csv` file. We will use `pandas` to take a look at some of the rows."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>serial</th>\n",
       "      <th>date</th>\n",
       "      <th>age</th>\n",
       "      <th>distance</th>\n",
       "      <th>stores</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>2009</td>\n",
       "      <td>21</td>\n",
       "      <td>9</td>\n",
       "      <td>6</td>\n",
       "      <td>84</td>\n",
       "      <td>121</td>\n",
       "      <td>14264</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2007</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>86</td>\n",
       "      <td>121</td>\n",
       "      <td>12032</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>2016</td>\n",
       "      <td>18</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>90</td>\n",
       "      <td>120</td>\n",
       "      <td>13560</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>2002</td>\n",
       "      <td>13</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>80</td>\n",
       "      <td>128</td>\n",
       "      <td>12029</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>2014</td>\n",
       "      <td>25</td>\n",
       "      <td>5</td>\n",
       "      <td>8</td>\n",
       "      <td>81</td>\n",
       "      <td>122</td>\n",
       "      <td>14157</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   serial  date  age  distance  stores  latitude  longitude  price\n",
       "0       0  2009   21         9       6        84        121  14264\n",
       "1       1  2007    4         2       3        86        121  12032\n",
       "2       2  2016   18         3       7        90        120  13560\n",
       "3       3  2002   13         2       2        80        128  12029\n",
       "4       4  2014   25         5       8        81        122  14157"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('data.csv', names = column_names)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.2: Check Missing Data\n",
    "\n",
    "It's a good practice to check if the data has any missing values. In real world data, this is quite common and must be taken care of before any data pre-processing or model training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "serial       0\n",
       "date         0\n",
       "age          0\n",
       "distance     0\n",
       "stores       0\n",
       "latitude     0\n",
       "longitude    0\n",
       "price        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task 3: Data Normalization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.1: Data Normalization\n",
    "\n",
    "We can make it easier for optimization algorithms to find minimas by normalizing the data before training a model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>age</th>\n",
       "      <th>distance</th>\n",
       "      <th>stores</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.015978</td>\n",
       "      <td>0.181384</td>\n",
       "      <td>1.257002</td>\n",
       "      <td>0.345224</td>\n",
       "      <td>-0.307212</td>\n",
       "      <td>-1.260799</td>\n",
       "      <td>0.350088</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.350485</td>\n",
       "      <td>-1.319118</td>\n",
       "      <td>-0.930610</td>\n",
       "      <td>-0.609312</td>\n",
       "      <td>0.325301</td>\n",
       "      <td>-1.260799</td>\n",
       "      <td>-1.836486</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.298598</td>\n",
       "      <td>-0.083410</td>\n",
       "      <td>-0.618094</td>\n",
       "      <td>0.663402</td>\n",
       "      <td>1.590328</td>\n",
       "      <td>-1.576456</td>\n",
       "      <td>-0.339584</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-1.266643</td>\n",
       "      <td>-0.524735</td>\n",
       "      <td>-0.930610</td>\n",
       "      <td>-0.927491</td>\n",
       "      <td>-1.572238</td>\n",
       "      <td>0.948803</td>\n",
       "      <td>-1.839425</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.932135</td>\n",
       "      <td>0.534444</td>\n",
       "      <td>0.006938</td>\n",
       "      <td>0.981581</td>\n",
       "      <td>-1.255981</td>\n",
       "      <td>-0.945141</td>\n",
       "      <td>0.245266</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       date       age  distance    stores  latitude  longitude     price\n",
       "0  0.015978  0.181384  1.257002  0.345224 -0.307212  -1.260799  0.350088\n",
       "1 -0.350485 -1.319118 -0.930610 -0.609312  0.325301  -1.260799 -1.836486\n",
       "2  1.298598 -0.083410 -0.618094  0.663402  1.590328  -1.576456 -0.339584\n",
       "3 -1.266643 -0.524735 -0.930610 -0.927491 -1.572238   0.948803 -1.839425\n",
       "4  0.932135  0.534444  0.006938  0.981581 -1.255981  -0.945141  0.245266"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.iloc[:, 1:]\n",
    "df_norm = (df - df.mean())/df.std()\n",
    "df_norm.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.2: Convert Label Value\n",
    "\n",
    "Because we are using normalized values for the labels, we will get the predictions back from a trained model in the same distribution. So, we need to convert the predicted values back to the original distribution if we want predicted prices."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "14263\n"
     ]
    }
   ],
   "source": [
    "y_mean = df['price'].mean()\n",
    "y_std = df['price'].std()\n",
    "\n",
    "def convert_label_value(pred):\n",
    "    return int(pred * y_std + y_mean)\n",
    "\n",
    "print(convert_label_value(0.350088))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task 4: Create Training and Test Sets"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.1: Select Features\n",
    "\n",
    "Make sure to remove the column __price__ from the list of features as it is the label and should not be used as a feature."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>age</th>\n",
       "      <th>distance</th>\n",
       "      <th>stores</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.015978</td>\n",
       "      <td>0.181384</td>\n",
       "      <td>1.257002</td>\n",
       "      <td>0.345224</td>\n",
       "      <td>-0.307212</td>\n",
       "      <td>-1.260799</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.350485</td>\n",
       "      <td>-1.319118</td>\n",
       "      <td>-0.930610</td>\n",
       "      <td>-0.609312</td>\n",
       "      <td>0.325301</td>\n",
       "      <td>-1.260799</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.298598</td>\n",
       "      <td>-0.083410</td>\n",
       "      <td>-0.618094</td>\n",
       "      <td>0.663402</td>\n",
       "      <td>1.590328</td>\n",
       "      <td>-1.576456</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-1.266643</td>\n",
       "      <td>-0.524735</td>\n",
       "      <td>-0.930610</td>\n",
       "      <td>-0.927491</td>\n",
       "      <td>-1.572238</td>\n",
       "      <td>0.948803</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.932135</td>\n",
       "      <td>0.534444</td>\n",
       "      <td>0.006938</td>\n",
       "      <td>0.981581</td>\n",
       "      <td>-1.255981</td>\n",
       "      <td>-0.945141</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       date       age  distance    stores  latitude  longitude\n",
       "0  0.015978  0.181384  1.257002  0.345224 -0.307212  -1.260799\n",
       "1 -0.350485 -1.319118 -0.930610 -0.609312  0.325301  -1.260799\n",
       "2  1.298598 -0.083410 -0.618094  0.663402  1.590328  -1.576456\n",
       "3 -1.266643 -0.524735 -0.930610 -0.927491 -1.572238   0.948803\n",
       "4  0.932135  0.534444  0.006938  0.981581 -1.255981  -0.945141"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = df_norm.iloc[:, :6]\n",
    "x.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.2: Select Labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.350088\n",
       "1   -1.836486\n",
       "2   -0.339584\n",
       "3   -1.839425\n",
       "4    0.245266\n",
       "Name: price, dtype: float64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = df_norm.iloc[:, -1]\n",
    "y.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.3: Feature and Label Values\n",
    "\n",
    "We will need to extract just the numeric values for the features and labels as the TensorFlow model will expect just numeric values as input."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "features array shape: (5000, 6)\n",
      "labels array shape: (5000,)\n"
     ]
    }
   ],
   "source": [
    "x_arr = x.values\n",
    "y_arr = y.values\n",
    "\n",
    "print('features array shape:', x_arr.shape)\n",
    "print('labels array shape:', y_arr.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.4: Train and Test Split\n",
    "\n",
    "We will keep some part of the data aside as a __test__ set. The model will not use this set during training and it will be used only for checking the performance of the model in trained and un-trained states. This way, we can make sure that we are going in the right direction with our model training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set: (4750, 6) (4750,)\n",
      "Test set: (250, 6) (250,)\n"
     ]
    }
   ],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.05, \n",
    "                                                    random_state=0)\n",
    "\n",
    "print('Training set:', x_train.shape, y_train.shape)\n",
    "print('Test set:', x_test.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task 5: Create the Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5.1: Create the Model\n",
    "\n",
    "Let's write a function that returns an untrained model of a certain architecture."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense (Dense)                (None, 10)                70        \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 20)                220       \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 5)                 105       \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 1)                 6         \n",
      "=================================================================\n",
      "Total params: 401\n",
      "Trainable params: 401\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "def get_model():\n",
    "    model = Sequential([\n",
    "        Dense(10, input_shape = (6,), activation='relu'),\n",
    "        Dense(20, activation='relu'),\n",
    "        Dense(5, activation='relu'),\n",
    "        Dense(1)\n",
    "    ])\n",
    "    model.compile(\n",
    "        loss='mse',\n",
    "        optimizer='adam'\n",
    "    )\n",
    "    return model\n",
    "get_model().summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task 6: Model Training"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6.1: Model Training\n",
    "\n",
    "We can use an `EarlyStopping` callback from Keras to stop the model training if the validation loss stops decreasing for a few epochs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 4750 samples, validate on 250 samples\n",
      "Epoch 1/100\n",
      "4750/4750 [==============================] - 2s 495us/sample - loss: 0.8461 - val_loss: 0.3754\n",
      "Epoch 2/100\n",
      "4750/4750 [==============================] - 0s 66us/sample - loss: 0.3233 - val_loss: 0.2107\n",
      "Epoch 3/100\n",
      "4750/4750 [==============================] - 0s 66us/sample - loss: 0.2115 - val_loss: 0.1809\n",
      "Epoch 4/100\n",
      "4750/4750 [==============================] - 0s 45us/sample - loss: 0.1822 - val_loss: 0.1698\n",
      "Epoch 5/100\n",
      "4750/4750 [==============================] - 0s 48us/sample - loss: 0.1705 - val_loss: 0.1624\n",
      "Epoch 6/100\n",
      "4750/4750 [==============================] - 0s 42us/sample - loss: 0.1650 - val_loss: 0.1589\n",
      "Epoch 7/100\n",
      "4750/4750 [==============================] - 0s 43us/sample - loss: 0.1619 - val_loss: 0.1518\n",
      "Epoch 8/100\n",
      "4750/4750 [==============================] - 0s 42us/sample - loss: 0.1598 - val_loss: 0.1521\n",
      "Epoch 9/100\n",
      "4750/4750 [==============================] - 0s 43us/sample - loss: 0.1583 - val_loss: 0.1501\n",
      "Epoch 10/100\n",
      "4750/4750 [==============================] - 0s 41us/sample - loss: 0.1571 - val_loss: 0.1517\n",
      "Epoch 11/100\n",
      "4750/4750 [==============================] - 0s 40us/sample - loss: 0.1573 - val_loss: 0.1480\n",
      "Epoch 12/100\n",
      "4750/4750 [==============================] - 0s 41us/sample - loss: 0.1558 - val_loss: 0.1473\n",
      "Epoch 13/100\n",
      "4750/4750 [==============================] - 0s 43us/sample - loss: 0.1549 - val_loss: 0.1516\n",
      "Epoch 14/100\n",
      "4750/4750 [==============================] - 0s 40us/sample - loss: 0.1546 - val_loss: 0.1466\n",
      "Epoch 15/100\n",
      "4750/4750 [==============================] - 0s 41us/sample - loss: 0.1538 - val_loss: 0.1520\n",
      "Epoch 16/100\n",
      "4750/4750 [==============================] - 0s 39us/sample - loss: 0.1547 - val_loss: 0.1481\n",
      "Epoch 17/100\n",
      "4750/4750 [==============================] - 0s 40us/sample - loss: 0.1535 - val_loss: 0.1464\n",
      "Epoch 18/100\n",
      "4750/4750 [==============================] - 0s 42us/sample - loss: 0.1532 - val_loss: 0.1497\n",
      "Epoch 19/100\n",
      "4750/4750 [==============================] - 0s 49us/sample - loss: 0.1524 - val_loss: 0.1484\n",
      "Epoch 20/100\n",
      "4750/4750 [==============================] - 0s 48us/sample - loss: 0.1516 - val_loss: 0.1503\n",
      "Epoch 21/100\n",
      "4750/4750 [==============================] - 0s 46us/sample - loss: 0.1524 - val_loss: 0.1457\n",
      "Epoch 22/100\n",
      "4750/4750 [==============================] - 0s 45us/sample - loss: 0.1518 - val_loss: 0.1493\n",
      "Epoch 23/100\n",
      "4750/4750 [==============================] - 0s 42us/sample - loss: 0.1516 - val_loss: 0.1502\n",
      "Epoch 24/100\n",
      "4750/4750 [==============================] - 0s 42us/sample - loss: 0.1514 - val_loss: 0.1501\n",
      "Epoch 25/100\n",
      "4750/4750 [==============================] - 0s 44us/sample - loss: 0.1507 - val_loss: 0.1481\n",
      "Epoch 26/100\n",
      "4750/4750 [==============================] - 0s 40us/sample - loss: 0.1517 - val_loss: 0.1468\n"
     ]
    }
   ],
   "source": [
    "es_cb = EarlyStopping(monitor='val_loss', patience=5)\n",
    "\n",
    "model = get_model()\n",
    "preds_on_untrained = model.predict(x_test)\n",
    "\n",
    "history = model.fit(\n",
    "    x_train, y_train,\n",
    "    validation_data = (x_test, y_test),\n",
    "    epochs = 100,\n",
    "    callbacks = [es_cb]\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6.2: Plot Training and Validation Loss\n",
    "\n",
    "Let's use the `plot_loss` helper function to take a look training and validation loss."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfUAAAHjCAYAAAA6x4aXAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3XmcXHWd7//3p/ZeqjtLd9JZSQeyECBsARTDMtfBCYIEkcsyggJyEUZFx4FrnOs4yMX5edWfMs6gMzjCjKMSGR2WcUCug4zAoJgEkgCBkBAC6ezpJJ3eu6vqe/84VZ1Kp7q7qrsqtfTr+XjU4yx1zulvVeqR9/l+z/d8jznnBAAAyp+v2AUAAAD5QagDAFAhCHUAACoEoQ4AQIUg1AEAqBCEOgAAFYJQBwCgQhDqAABUCEIdAIAKESh2AXLV0NDg5syZU+xiAABwTKxZs2afc64xm23LLtTnzJmj1atXF7sYAAAcE2b2Trbb0vwOAECFINQBAKgQhDoAABWi7K6pAwCG19/fr5aWFvX09BS7KMhBJBLRzJkzFQwGR30MQh0AKkxLS4ui0ajmzJkjMyt2cZAF55xaW1vV0tKi5ubmUR+H5ncAqDA9PT2aPHkygV5GzEyTJ08ec+sKoQ4AFYhALz/5+Dcj1AEAqBCEOgAgry688EI99dRTR6y799579Sd/8ifD7ldbWytJ2rFjh6688sohjz3SAGT33nuvurq6BpY/+MEP6uDBg9kUfVh33XWXvvnNb475OIVEqAMA8uraa6/VypUrj1i3cuVKXXvttVntP336dP3sZz8b9d8fHOpPPPGEJkyYMOrjlRN6vwNABfvKv72mDTsO5fWYi6bX6S8/dNKQ71955ZX60pe+pN7eXoXDYW3dulU7duzQ0qVL1dHRoeXLl+vAgQPq7+/XPffco+XLlx+x/9atW3XppZfq1VdfVXd3t2688UZt2LBBJ554orq7uwe2u+2227Rq1Sp1d3fryiuv1Fe+8hV95zvf0Y4dO/QHf/AHamho0DPPPDMwvHhDQ4O+9a1v6YEHHpAk3Xzzzfrc5z6nrVu36uKLL9bSpUv1wgsvaMaMGXrsscdUVVWV1feR6ZidnZ266qqr1NLSong8rr/4i7/Q1VdfrRUrVujxxx9XIBDQBz7wgbzX/Al1AEBeTZ48WWeffbZ++ctfavny5Vq5cqWuvvpqmZkikYgeeeQR1dXVad++fXrPe96jyy67bMhOYt/73vdUXV2t9evXa/369TrjjDMG3vvqV7+qSZMmKR6P6/3vf7/Wr1+v22+/Xd/61rf0zDPPqKGh4YhjrVmzRg8++KBefPFFOed0zjnn6IILLtDEiRO1adMmPfTQQ/r+97+vq666Sj//+c913XXXjfhZhzrmli1bNH36dP37v/+7JKmtrU379+/XI488ojfeeENmlpdLAoMR6gBQwYarURdSqgk+FeqpmqxzTn/+53+uZ599Vj6fT9u3b9fu3bvV1NSU8TjPPvusbr/9dknS4sWLtXjx4oH3Hn74Yd1///2KxWLauXOnNmzYcMT7gz3//PP68Ic/rJqaGknSFVdcoeeee06XXXaZmpubddppp0mSzjzzTG3dujWrzznUMZctW6Y77rhDX/jCF3TppZfqvPPOUywWUyQS0c0336xLLrlEl156aVZ/IxdcUwcA5N3ll1+up59+Wi+99JK6u7sHatg//vGPtXfvXq1Zs0Zr167V1KlTR7w3O1Mt/u2339Y3v/lNPf3001q/fr0uueSSEY/jnBvyvXA4PDDv9/sVi8WGPdZIx5w/f77WrFmjU045RV/84hd19913KxAI6Pe//70+8pGP6NFHH9WyZcuy+hu5INQBAHlXW1urCy+8UDfddNMRHeTa2to0ZcoUBYNBPfPMM3rnneGfKnr++efrxz/+sSTp1Vdf1fr16yVJhw4dUk1Njerr67V79249+eSTA/tEo1G1t7dnPNajjz6qrq4udXZ26pFHHtF55503ps851DF37Nih6upqXXfddbrjjjv00ksvqaOjQ21tbfrgBz+oe++9V2vXrh3T386E5ncAQEFce+21uuKKK47oCf/Rj35UH/rQh7RkyRKddtppWrhw4bDHuO2223TjjTdq8eLFOu2003T22WdLkk499VSdfvrpOumkkzR37ly9733vG9jnlltu0cUXX6xp06bpmWeeGVh/xhln6IYbbhg4xs0336zTTz8966Z2Sbrnnnt07733Diy3tLRkPOZTTz2lO++8Uz6fT8FgUN/73vfU3t6u5cuXq6enR845ffvb387672bLhmuOKEVLlixxI92jmLV4TOo9JEUmSD4aLQBUhtdff10nnnhisYuBUcj0b2dma5xzS7LZf3wn2cv/LH29WerYVeySAAAwZuM71MNRb9p79LUXAADKzTgP9TpvSqgDACrAOA/1VE09v6MtAQBQDIS6RE0dAFARCHWJUAcAVARCXSLUASCPWltbddppp+m0005TU1OTZsyYMbDc19eX1TFuvPFGbdy4cdht7rvvvoGBacZq6dKlBRkM5lgb34PPpEK9h2vqAJAvkydPHgjIu+66S7W1tbrjjjuO2MY5J+ecfEOMEfLggw+O+Hc+9alPjb2wFWZ8h7o/KAWq6CgHoHI9uULa9Up+j9l0inTx13LebfPmzbr88su1dOlSvfjii/rFL36hr3zlKwPjw1999dX68pe/LMmrOf/t3/6tTj75ZDU0NOjWW2/Vk08+qerqaj322GOaMmWKvvSlL6mhoUGf+9zntHTpUi1dulS//vWv1dbWpgcffFDnnnuuOjs79bGPfUybN2/WokWLtGnTJv3DP/zDwMNbhtPd3a1bb71VL730koLBoO69916df/75euWVV3TTTTepv79fiURCjz76qBobG3XVVVdpx44disfjuuuuu3TllVfm/B2N1fhufpekSB3N7wBwjGzYsEGf+MQn9PLLL2vGjBn62te+ptWrV2vdunX61a9+pQ0bNhy1T1tbmy644AKtW7dO733vewee+DaYc06///3v9Y1vfEN33323JOlv/uZv1NTUpHXr1mnFihV6+eWXsy7rd77zHYVCIb3yyiv653/+Z11//fXq6+vTd7/7Xd1xxx1au3atVq1apenTp+uJJ57QnDlztG7dOr366qu66KKLRvcFjdH4rqlLXhM8oQ6gUo2iRl1Ixx9/vM4666yB5Yceekg/+MEPFIvFtGPHDm3YsEGLFi06Yp+qqipdfPHFkrzHoj733HMZj33FFVcMbJMaz/3555/XF77wBUneePEnnZT9o2iff/553XnnnZKkk046SdOnT9fmzZt17rnn6p577tE777yjK664QieccIIWL16sFStWaMWKFfrQhz50xFj0xxI1dUIdAI6Z1HPHJWnTpk3667/+a/3617/W+vXrtWzZsoyPTw2FQgPzwz0WNfX41PRtxvJ8k6H2vf766/XII48oHA7roosu0rPPPqsTTzxRq1ev1kknnaQ777xTf/VXfzXqvzsWhDqhDgBFcejQIUWjUdXV1Wnnzp166qmn8v43li5dqocffliS9Morr2Rs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\n",
      "text/plain": [
       "<Figure size 576x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_loss(history)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task 7: Predictions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7.1: Plot Raw Predictions\n",
    "\n",
    "Let's use the `compare_predictions` helper function to compare predictions from the model when it was untrained and when it was trained."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfQAAAHjCAYAAADVBe2pAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzs3Xl8VNXZwPHfTUgkIxAkWBdsErVaiwYQcUcR0VfEIkqlaocQcUnFDeyitfNWltdp1fZVsHWLokYYrUtxwRdXFHFpq6hgLFq3ZiLiAkHCkkBIct4/biaZ5d6Ze2fu7M/38+ETM5m5c2aIPHPOec7zaEophBBCCJHdCtI9ACGEEEIkTgK6EEIIkQMkoAshhBA5QAK6EEIIkQMkoAshhBA5QAK6EEIIkQMkoAshhBA5QAK6EEIIkQMkoAshhBA5oE+6B2DH4MGDVWVlZbqHIYRIt3feMf/ZEUekbhwip72z3vz37Ih9k/t7tmkT/Oc/AO9sVErtaeUxWRXQKysrWbVqVbqHIYRIt8pK8Psjb6+oAPk3Qjikcn4l/pbI37OK0gpWzUre79natVBVBSecAK+9phn8ohuTJXchRPbxesHlCr3N5dJvF8Ih3nFeXEWhv2euIhfeccn9PRs6FB58EJ591t7jJKALIbKP2w11dfqMXNP0r3V1+u1COMRd5aZuYh0VpRVoaFSUVlA3sQ53VXJ+z+65p3eBye2G3Xe393gtm7qtjRo1SsmSuxBCiFyzYAHMmgXV1frsPEDTtHeUUqOsXCOr9tCN7Nq1i3Xr1rFjx450D0U4qG/fvuy3334UFRWleyhCCJFUN98M114LkyfDvffGf52sD+jr1q2jf//+VFZWomlauocjHKCUorm5mXXr1rH//vunezhCCJE0//M/cP31cN55+sw8kTlM1u+h79ixg7KyMgnmOUTTNMrKymTVRQiR0zo79T3z6mpYvDixYA45MEMHJJjnIPk7FULkKqVg2zbo3x8eewwKC/U/icr6GboQQgiRLZSCq6+G0aNh61YoLnYmmEM+BnSfTy9KUVCgf/X5ErpcY2Mjhx12WMhtc+bM4U9/+lPUx61evZply5bZfr7169dzzjnn2H6ckRUrVvDjH//Y8HZN01i4cGHPbe+99x6apsV8XcGM3pt47iOEELmgqwsuv1zPaD/5ZOjXz9nr51dA9/mgtlavMKWU/rW2NuGgHo9oAb2jo8P0cfvuuy+PP/54sobVo6qqikceeaTn+7/+9a8MHz486c8rhBC5qLNTDzd33gnXXAO33KKXUHBSfgV0jwdaW0Nva23Vb0+Sk046iWuvvZajjjqKgw8+mNdee4329nauv/56HnnkEUaMGMEjjzzCnDlzqK2t5b/+67+YNm0ajY2NnHDCCYwcOZKRI0fy5ptvAqEz2gceeIDJkyczfvx4DjroIK655pqe533hhRc49thjGTlyJFOmTGHbtm0APPfccxxyyCGMHj2aJUuWmI67vLycHTt28M0336CU4rnnnuP000/v+fnq1as55phjGDZsGGeffTbfffcdAO+88w7Dhw/n2GOP5fbbb++5f2dnJ7/+9a858sgjGTZsGHfffbdzb7IQQmS466+HhQvhd7+DG290PphDvgX0piZ7tzuko6ODt956i/nz5zN37lyKi4uZN28e5557LqtXr+bcc88F9GD41FNP8dBDD/G9732PF198kXfffZdHHnmEq666yvDaq1ev5pFHHqGhoYFHHnmEL774go0bN3LDDTfw0ksv8e677zJq1ChuueUWduzYwSWXXMLSpUt57bXX+Prrr6OO+5xzzuGxxx7jzTffZOTIkey22249P5s2bRo33XQT77//PlVVVcydOxeA6dOnc9ttt/H3v/895FoLFy6ktLSUt99+m7fffpt77rmH/+idB4QQIudddhn8+c8wb15ygjnkW0AvL7d3uwVm2djBt0+ePBmAI444gsbGRtNrnXnmmZSUlAB6wZxLLrmEqqoqpkyZwtq1aw0fM27cOEpLS+nbty9Dhw7F7/fzj3/8g7Vr13L88cczYsQI6uvr8fv9fPTRR+y///4cdNBBaJrG1KlTo762n/70pzz22GM8/PDDnH/++T23t7S0sHnzZsaMGQNATU0NK1eujLi9urq65zEvvPACDz74ICNGjODoo4+mubmZTz75JOrzCyFENmtvh9tu05fbhwyBK65I7vPlxLE1y7xefRMjeNk9wYYOZWVlPcvNAZs2bQopiBKY2RYWFkbdH989qHDvrbfeyl577cWaNWvo6uqib9++ho8JnjUHrq+U4tRTT+Xhhx8Oue/q1attHQfbe++9KSoq4sUXX2TBggU9y/5mlFKm11dK8ec//5nTTjst5PZoH3CEECJb7dwJU6bA0qXwwx9C2D99SZFfM/QkNHTo168f++yzD8uXLwf0YP7cc88xevToqI/r378/W7duNf15S0sL++yzDwUFBSxatIjOzk7LYzrmmGN44403+PTTTwFobW3l448/5pBDDuE///kPn332GUBEwDcyb948brrpJgqDzlWUlpayxx578NprrwGwaNEixowZw8CBAyktLeX1118HwBeUbHjaaadx5513smvXLgA+/vhjtm/fbvk1CSFEtmhrg7PO0oP5HXekJphDvs3QQQ/eDndkevDBB7n88sv55S9/CcDs2bM58MADoz5m7Nix3HjjjYwYMYLrrrsu4ueXXXYZP/nJT3jssccYO3ZsyOw9lj333JMHHniA888/n507dwJwww03cPDBB1NXV8cZZ5zB4MGDGT16NB988EHUax133HGGt9fX13PppZfS2trKAQccwP333w/A/fffz4UXXojL5QqZjV988cU0NjYycuRIlFLsueeePPnkk5ZfkxBCZIPt22HSJHj5Zb0u+0UXpe65s77b2ocffsiPfvSjNI1IJJP83Qohss2778JJJ8Ff/gLTpiV+vbzqtiaEEEKk265dei32kSPh889h8ODUjyG/9tCFEEIIh333HRx/vF40BtITzEFm6EIIIUTcNm6EU0+FtWthv/3SOxYJ6EIIIUQcvv0WTjkFPvkEnnoKxo9P73hkyV0IYY/DDY6ESBZfg4/K+ZUUzC2gcn4lvgbnflfb2vTkt08/hWeeSX8wB5mhCyHsCDQ4ChRnCjQ4AsePgwqRCF+Dj9qltbTu0n9X/S1+apfqv6vuqsR/V0tK4NJLYfhw6C6OmXZ5N0N3+hNbc3MzI0aMYMSIEey9994MGTKk5/v29nZL15g+fTr//ve/ExpHwH777cfmzZsNbx87dmzIbYcddhgjRoywdf3Ro0ezevXqhO8jslQaGhwJEQ/Pck9PMA9o3dWKZ3liv6t+PwRaVVx1VeYEc8izGXoyPrGVlZX1BK85c+bQr18/fvWrX4XcRymFUoqCAuPPT4GiLMm2efNm1q9fz7777ktDQwN9+uTVX79wQpoaHAlhV1OL8e+k2e1WfPaZ3sdc0+Djj6G4OO5LJUVezdCT9YnNyKeffsphhx3GpZdeysiRI/nqq6+ora1l1KhRHHroocybN6/nvoEZbUdHBwMHDuQ3v/lN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\n",
      "text/plain": [
       "<Figure size 576x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "preds_on_trained = model.predict(x_test)\n",
    "compare_predictions(preds_on_untrained, preds_on_trained, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7.2: Plot Price Predictions\n",
    "\n",
    "The plot for price predictions and raw predictions will look the same with just one difference: The x and y axis scale is changed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgQAAAHjCAYAAABcqwcxAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzs3Xl4VOXZ+PHvk5AIUQgQcMUkarFWDVJAW1c2V1yhWrQDxjUudcHWpe30VbDm/WnrW8AFNRYUYaqodaOCCnHBpS6IQBQ31EykuECAsCQQkjy/P85MmOWcmTMzZ9bcn+viSnJm5pwzk+i5z/Pcz30rrTVCCCGE6Nry0n0CQgghhEg/CQiEEEIIIQGBEEIIISQgEEIIIQQSEAghhBACCQiEEEIIgQQEQgghhEACAiGEEEIgAYEQQgghgG7pPoFU69evny4vL0/3aQgh0u3DD60fGzo0dechctqHa63/zobum9y/s2+/hR9/BPhwvda6f7Tnd7mAoLy8nKVLl6b7NIQQ6VZeDl5v+PayMpD/RwiHlE8rx9sU/ndWVlzG0knJ+zt76CG48kq44QaYOlWZ/KGHkykDIUTXVF0NRUXB24qKjO1COKR6dDVFBcF/Z0UFRVSPTu7f2UUXwaOPwv/9n/3XSEAghOiaXC6oqTFGBJQyvtbUGNuFcIirwkXNmTWUFZehUJQVl1FzZg2uCuf/ztra4NZbobERdtsNKiuNP227VFfrdjhs2DAtUwZCCCFyyc6dRiz71FMwcyZccsmux5RSH2qth0XbR5fLITCzc+dO1qxZw/bt29N9KsIh3bt3Z8CAARQUFKT7VIQQIql27IDzz4fnnoO77w4OBmIhAQGwZs0aevbsSXl5OSqW8RWRkbTWNDY2smbNGg444IB0n44QQiTN9u3wq1/BggVwzz1w7bXx70tyCIDt27dTUlIiwUCOUEpRUlIiIz5CiJy3aRN8+aWxqiCRYABkhKCTBAO5RX6fQohc1txsJA7uvTesXAnduye+TxkhEEIIIbLI5s1wyilw+eXGz04EAyABQXw8HqOoSV6e8dXjSWh39fX1HH744UHbJk+ezN133x3xdcuXL2fBggUxH2/t2rWce+65Mb/OzOuvv84ZZ5xhul0pxcyZMzu3ffTRRyilor6vQGafTTzPEUKIXLBpE5x8Mrz7Lpx2mrP7loAgVh4PVFUZFc60Nr5WVSUcFMQjUkDQ1tZm+bp9992Xp59+Olmn1amiooJ58+Z1/vzEE09wxBFHJP24QgiRizZsgBNPhGXLjOWF553n7P4lIIiV221M3gRqbja2J8mIESO45ZZbOOqoozj44IN58803aW1t5dZbb2XevHkMHjyYefPmMXnyZKqqqjj55JO58MILqa+v5/jjj2fIkCEMGTKEd955Bwi+o3700UcZN24cp556KgMHDuTmm2/uPO4rr7zC0UcfzZAhQzjvvPPYunUrAC+99BKHHHIIxx13HM8884zleZeWlrJ9+3Z++OEHtNa89NJLnBYQ0i5fvpxf/vKXDBo0iLFjx7Jx40YAPvzwQ4444giOPvpo7r///s7nt7e3c9NNN3HkkUcyaNAgHnroIec+ZCGEyGBaw5lnwscfw7PPwjnnOH8MCQhi1dAQ23aHtLW18f777zNt2jSmTJlCYWEht99+O+PHj2f58uWMHz8eMC6mzz//PP/85z/Zc889WbRoEcuWLWPevHlcd911pvtevnw58+bNo66ujnnz5vHtt9+yfv167rjjDhYvXsyyZcsYNmwYf//739m+fTuXX3458+fP58033+T777+PeN7nnnsuTz31FO+88w5Dhgxht91263zswgsv5K677mLlypVUVFQwZcoUAC6++GLuuece/vOf/wTta+bMmRQXF/PBBx/wwQcf8PDDD/PNN98k8rEKIURWUAruuANeeAFOPz05x5BVBrEqLTVviFJaGvcurTLiA7ePGzcOgKFDh1JfX2+5r7POOosePXoARsGla665huXLl5Ofn88XX3xh+prRo0dTXFwMwKGHHorX62XTpk2sWrWKY489FoDW1laOPvpoPvvsMw444AAGDhwIwIQJE6ipqbE8n1//+teMHz+ezz77jAsuuKBzlKKpqYlNmzYxfPhwACorKznvvPPCtk+cOJGFCxcCxojFypUrO6c7mpqa+PLLLzn44IMtjy+EENnsv/+F1183qhCOHJncY0lAEKvqaiNnIHDaIMGGKCUlJZ3D5X4bNmwIKqrjv7POz8+PmB+w++67d34/depU9tprL1asWEFHRwfdLVJRA+/a/fvXWnPSSSfx+OOPBz13+fLlMS3p23vvvSkoKGDRokVMnz69MyCworW23L/WmnvvvZdTTjklaHukAEkIIbJVQwOMGgXr1hmJhP2jNjBOjEwZxCoJDVH22GMP9tlnH2prawEjGHjppZc47rjjIr6uZ8+ebNmyxfLxpqYm9tlnH/Ly8pgzZw7t7e22z+mXv/wlb7/9NqtXrwagubmZL774gkMOOYRvvvmGr776CiAsYDBz++23c9ddd5Gfn9+5rbi4mD59+vDmm28CMGfOHIYPH07v3r0pLi7mrbfeAsATkKx5yimn8MADD7Bz504AvvjiC7Zt22b7PQkhRLb45hsYPhzWr4eXX05+MAAyQhAfl8vxjmiPPfYYv/3tb/n9738PwG233cZBBx0U8TUjR47kzjvvZPDgwfzxj38Me/zqq6/mV7/6FU899RQjR44MGj2Ipn///jz66KNccMEF7NixA4A77riDgw8+mJqaGk4//XT69evHcccdx8cffxxxX8ccc4zp9tmzZ3PllVfS3NzMgQceyCOPPALAI488wiWXXEJRUVHQaMBll11GfX09Q4YMQWtN//79ee6552y/JyGEyAarVxsjA1u3wuLFMCxqWyJnSLdD4NNPP+VnP/tZms5IJIv8XoUQ2eihh+DPf4ZFi2Dw4MT3Z7fboUwZCCGEEBnANxvKFVfAp586EwzEQgICIYQQIs2WL4dDDoH33zd+7tcv9ecgAYEQQgiRRkuXGjkDO3dC377pOw8JCIQQQog0efddGD0aiothyRL4yU/Sdy4SEAghUsvh5mBCJJOnzkP5tHLypuRRPq0cT51zf68ffwwnnWQsKVyyxPjPIZ0kIBBCpE4GNQcTIhpPnYeq+VV4m7xoNN4mL1XzqxwLCn76U6OF8RtvwP77O7LLhEhAEAenI8bGxkYGDx7M4MGD2Xvvvdlvv/06f25tbbW1j4svvpjPP/88ofPwGzBgAJs2bTLdPjKkdubhhx/O4BhTYY877jiWL1+e8HNEFkpDczAh4uWuddO8M/jvtXlnM+7axP5eX38dfvgBCgrg73+H/fZLaHeOkcJEMfJHjP4/En/ECOCqiK9YUUlJSefFb/Lkyeyxxx7ceOONQc/RWqO1Ji/PPIbzF/VJtk2bNrF27Vr23Xdf6urq6NZN/oREDNLUHEyIeDQ0mf9dWm2348UXYdw4+NWv4J//jHs3SSEjBDFKVsRoZvXq1Rx++OFceeWVDBkyhO+++46qqiqGDRvGYYcdxu233975XP8ddVtbG7179+YPf/hDZwvhH3/8EYAffviBcePGMWzYMI466ijeffddANatW8dJJ53EkCFDuOqqq4hUrOq8887jySefBIyyxRdccEHnYy0tLVRWVlJRUcGQIUNYsmSJ8fk0N3PeeecxaNAgzj//fLZv3975moULF3a2WB4/fryUIs51Vk3AEmgOJkSylBab/11abY/m2Wdh7FgYNAjuuy+RM0sOCQhilIyIMZJVq1Zx6aWX8tFHH7Hffvtx5513snTpUlasWMGiRYtYtWpV2GuampoYPnw4K1as4Oijj2bWrFkAXHfdddx8880sXbqUJ598kssuuwwwyiSPHDmSZcuWceqpp7J27VrL8znvvPM6uw0uWLCA0wP6cN5zzz0UFhZSV1fHnDlzmDhxIq2trdx333306dOHlStXcsstt/DRRx8B8OOPP3LnnXdSW1vLsmXLGDRoENOnT3fssxMZqLraaAYWKMHmYEIkS/XoaooKgv9eiwqKqB4d+9/rk0/CeefB0KFGOeJ0Li+0IuO9MSotLsXbFN7+ON6IMZqDDjqII488svPnxx9/nJkzZ9LW1sbatWtZtWoVhx56aNBrevTowWmnnQYY7ZL9DYQWL14clGewceNGWlpaWLJkCQsWLADg7LPPpmfPnpbn079/f3bffXeeeOIJBg0aFNRB8a233uKmm24C4LDDDmPfffdl9erVLFmyhJtvvhmAn//85xx22GEAvPPOO6xataqz10Fra2vUhk4iy/l7gLjdxjRBaakRDDjcG0QIJ/ingd21bhqaGigtLqV6dHXM08NtbfCXv8DRR8OCBRDhf7FpJQFBjKpHVwflEED8EaMdgQ2JvvzyS6ZPn877779P7969mTBhQtDwu19hYWHn94HtkrXWvP/++0GP+8XS0nj8+PH89re/Ze7cuUHbI001mO1fa82pp57KnDlzbB9b5IAkNAcTIllcFa6488PAWEzTrZvRl6BnT4ihx1zKyZRBjFwVLmrOrKGsuAyFoqy4jJozaxL6g7Fr8+bN9OzZk169evHdd9/x8ssvx/T6E088kfvvv7/zZ38i4wknnNDZZnj+/PkRWyoD/OpXv+Lmm2/mpJNOCtoeuJ9PP/2U7777jp/85CdB21esWMEnn3wCGF0Q33jjDb7++msAtm3bxpdffhnTexJCiEz10EPwm98YIwR7753ZwQBIQBAXV4WL+kn1dNzWQf2k+pQEAwBDhgzh0EMP5fDDD+fyyy/n2GOPjen1999/P2+//TaDBg3i0EMP5eGHHwZgypQpLF68mCFDhvD666+zX5Q1MMXFxdxyyy1hKwyuvfZaWlpaqKiowOVy8dhjj1FYWMg111xDY2MjgwYNYurUqQzz9fLca6+9mDlzJuPHj+eII47gmGOO4YsvvojpPQkhRCa691648krYssUICLKBtD9G2uTmKvm9CiHS4e674aab4JxzYN48MJmlTSlpfyyEEEKkmD8Y+PWvjZUF6Q4GYiEBgRBCCOGQo4/eVY27oCDdZxMbCQh8utrUSa6T36cQIlW0hrfeMr4/9lgjmTAbi7hKQAB0796dxsZGuYjkCK01jY2NQTUShBAiGbSG3/8ejj8eXnst3WeTmCyMYZw3YMAA1qxZw7p169J9KsIh3bt3Z8CAAek+DSFEDuvogOuug/vvh2uvhREj0n1GiZGAACgoKOCAAw5I92kIIYTIEh0dcMUV8I9/wI03wl//CjHUd8tIMmUghBBCxOi114xgwO3OjWAAZIRACCGEiNno0fDuu/CLX6T7TJwjIwRCCCGEDTt3QmUl+PrF5VQwABIQCCGEEFHt2AHnnguPPQa+NjA5R6YMhBBCiAi2b4dx42DhQrjvPvjtb9N9RskhAYEQQghhoaUFzj4bFi+Gmhq4/PJ0n1HyJG3KQCk1Syn1o1Lq45Dt1yqlPldKfaKU+mvA9j8qpVb7HjslYPupvm2rlVJ/CNh+gFLqPaXUl0qpeUqpLKoYLYQQIhsUFhqtix95JLeDAUhuDsGjwKmBG5RSI4GzgUFa68OAu33bDwXOBw7zvWaGUipfKZUP3A+cBhwKXOB7LsBdwFSt9UBgI3BpEt+LEKIr83igvBzy8oyvHk+6z0gAnjoP5dPKyZuSR/m0cjx1zv1eNm+GtWshPx9mzzaSCXNd0gICrfUSYEPI5quAO7XWO3zP+dG3/WzgCa31Dq31N8Bq4Cjfv9Va66+11q3AE8DZSikFjAKe9r1+NnBOst6LEKIL83iMbjVer1Gn1uvd1b1GpI2nzkPV/Cq8TV40Gm+Tl6r5VY4EBZs2wcknG//a2nKjxoAdqV5lcDBwvG+o/w2l1JG+7fsB3wY8b41vm9X2EmCT1rotZLsppVSVUmqpUmqplCcWQsTE7Ybm5uBtzc3GdpE27lo3zTuDfy/NO5tx1yb2e2lsNGoMLFsG//u/2dmkKF6pDgi6AX2AXwI3AU/67vbN4i8dx3ZTWusarfUwrfWw/v37x37WQoiuq6Ehtu0iJRqazD9/q+12rFsHo0bBJ5/Ac8/BWWfFvauslOqAYA3wjDa8D3QA/Xzb9w943gBgbYTt64HeSqluIduFEF1Vsub5S0tj257DkjlnH6vSYvPP32q7HddeC19+CfPnw5gxce8ma6U6IHgOY+4fpdTBQCHGxf0F4Hyl1G5KqQOAgcD7wAfAQN+KgkKMxMMXtNGn+DXgXN9+K4HnU/pOhBCZI5nz/NXVUFQUvK2oyNjehSRzzj4e1aOrKSoI/r0UFRRRPTr+38s998CiRXDSSYmeXXZK5rLDx4H/AD9VSq1RSl0KzAIO9C1FfAKo9I0WfAI8CawCXgJ+q7Vu9+UIXAO8DHwKPOl7LsAtwO+UUqsxcgpmJuu9CCEyXDLn+V0uYwF6WZmRXVZWZvzsciW+7yySrDn7eLkqXNScWUNZcRkKRVlxGTVn1uCqiO330tAA11wDra2w555w7LFJOuEsoIyb7a5j2LBheunSpek+DSGEk/LyjJGBUEoZfWpFwvKm5KFNUrUUio7bsvMz/vprI2dg0yb4z3/gZz9L9xklh1LqQ631sGjPk14GQojsJ/P8SZeMOft0+vJLGD4ctmyBV1/N3WAgFhIQCCGyn8zzJ10y5uzT5dNPjWBg+3YjGBgyJN1nlBkkIBBCZD+Z5086p+bsM8G2bdCrF7z+OhxxRLrPJnNIDoEQQogu4YcfYK+9jO/b242yxF2B5BAIIbKP9AwQJpyof/DBB3DIITBjhvFzVwkGYtGFijIKITKav5aAf/mgv5YAyNB/F+avf+Bf8uivfwDYnq545x047TQoKYHTT0/aqWY9mTIQQmSG8nIjCAhVVgb19ak+G5EhyqeV420K/7soKy6jflJ91NcvWWJUHdx3XyOBcMCAJJxkhpMpAyFEdpGeAcJEIj0LfvjBCAb23x/eeKNrBgOxkIBACJEZ0lFLoAvmLGRSPwI7Eql/sNdeMHOmsZpgn30cPrEcJAGBECIzpLqWQDL7H2SoTOtHYEc89Q/+/W9YvNj4fvz4XSsLRGQSEAghMkOqawkks/9Bhsq0fgR2xFr/4JlnYOxYuOMO82rWwpokFQohuqZM7X/g8RhBSUODMV1SXe1YUBRLPwJPnQd3rZuGpgZKi0upHl2d8UWInngCJkyAo46ChQuhuDjdZ5QZJKlQCCEiycT+B0mexrA7H5+NUwuPPWbETcceCy+/LMFAPCQgEEJ0TZnY/yDJ0xh25+OzcWphyRIYMQIWLICePdN9NtlJAgIhRPZwclVAJvY/SNLSS//KgonPTKRHtx6U9CiJOB+fyFK/VNu2zfj60EPw4ouw++7pPZ9sJgGBEMJZyVrKl4zhdJfLKHrU0WF8TXdFRIenMTx1Hvr9tR8TnpnQOfzf2NJIS1sLc8bNoX5SvWleQLa0Op4+HQYNgu+/N0oRd++e7jPKbhIQCCGck8w58K6wKsDBaQx/HkBjS2PYY9GG/7Oh1fHf/gaTJsHgwdC3b7rPJjdIQCCEcE4yL9pdoZKhg9MYZnkAgSIN/2d6q+M77oCbb4bzzzdWFhQWpvuMcoMsOxRCOCeZS/mk10FMrJYY+tntBZBpamrgiitg4kSYNQu6SYu+qGTZoRAi9ZK5lC8TVwVksEjz/Zk2/B+Lc8+Fv/wFHnlEggGnSUAghHBOMi/asQ6nd8E+BYHM8gAASnqUpHX4P55eClobv+rt2418gT//2UgiFM6S+EoI4Rz/xTlJlfZwuezty5/c6M9n8Cc3Bp5jjvNf8DOp2qA/0dGf2+AveBR4vqE6OuDaa2HGDCO2u+yylJ1ulyM5BEKI3CP5BhmpfFo53qbw34tVPkNHh5Ev8I9/GEmEd95pDA6J2EgOgRCi6+oKKxKyUCwFj9rb4eKLjWDgz3+WYCAVJCAQQuSeTOxTIGIqeOT1Gm2Mb7/dSCKUYCD5JCAQQuQeWZGQkewUPGpvN5IIDzwQVq2C//mfVJ9l1yUBgRAi92RinwIRteDRjh0wdqwxKgCw115pPNkuSJIKhRBCpF1LC4wbBy+9BPffD1dfne4zyh12kwpl2aEQQoi0am6Gs86CV1+Fhx+WpYXpIlMGQghhposXNkqVjg44cuT31L7ajj67kju22itWJJwnIwRCCBFKChulzOOfeFi9/xuw/xaoeAJvE1GLFYnkkBwCIYQIJYWNkm7jRlixAi5aHluxIhE7ySEQQoh4SWGjpFq/Hk4+Gb76CjZftRl6hD8nUntmkRySQyCEEKGSVdhI8hL48UcYNcqoMTBvHpTt3cv0eaHFiuJpiiRiIwGBEEKESkZhI39egtdrVN7x5yV0oaDgu+9gxAhYvRpefBFOPdVesSJ/UyRvkxeN7myKJEGBsyQgEEKIUMkobOR270pS9GtuNrZ3Ef/4hzHrsnAhjB5tbItWrAiMjo3+Dol+zTubcdd2nc8uFSSpUAghUiEvzxgZCKWUsfYuh2m9622uXg0HHxzb6/Om5KEJ/+wUio7bcvuzc4J0OxRCiEzSRRsuffUVHH88fPONERPFGgxAbE2RRPwkIBBCiFTogg2XPv8chg+Hzz6Dpqb492Mnz0AkTgICIUTyJCOrPlsz9btYw6VVq4xgoLUVXnsNBg+Obz+eOk9nDkG+ygcwzTMwe52sSoiNBARCiOSwm1UfywU+0Uz9FAQTES9ELpdR2Kijw/iao8HAp58aqwmUgtdfh4qK4MftXqwDVxcAtOv2zpGBaMGArEqInSQVCiGSw061v9ASwWAMo1vdOSdSQTDWY8XBfyEKzIgvKiiKejeba5qa4OKL4c47w3MGYvmMyqfFV8Uw3tflKrtJhRIQCCGSw05WfawX+EQy9VNQjrirX4hWroSf/CQ8VSJQLJ9RvKsLZFVCMFllIIRILztZ9bGWCE4kUz8F5Yityu12hTK8b78Nxx0H110X+XmxfEbxri6QVQnxkYBACJEcdrLqY73AJ5Kpn4Jlf131QvTGG3DKKbDPPjB5cuTnxvIZxbu6QFYlxEcCAiFEctjJqo/1Ap9Ipn4Klv11xQvR4sVw2mnGr+L112HAgMjPj+UzslPF0Ey8r+vytNZd6t/QoUO1ECKDzJ2rdVmZ1koZX6+6KvjnuXOTd6zAfUd6LJZDrJyry6aWaSaj86fkayajy6aW6bkr55o+T01Wpo9ng+3bjY+qokLrH36w/7pceO/ZBFiqbVwfJalQCJE5UrASIBXHjZZJn0urET79FPbcE0pK0n0mwoqsMrAgAYEQGSwFKwFScdxomfTZvhrhX/+CZcvgjjuMmRuR2WSVgRAi+6RgJUAqjhstkz6bVyM8/jiMH2/kC+zYke6zEU6SgEAIkTnS1QAonuNGqHoYLZM+U1YjxFred/ZsmDABjj0WXnoJundP0YmKlJCAQAiROdLVACjW40YpoRwtkz4TViPEWt535kyj+uDIkbBgAfTsmbJTFSkiAYEQInOkqwFQrMd1u4MTEMH42e02dhdl2VsmLIvzNwwKegs7m3HXuk2f37MnnH46zJ8Pu++eijMUqSZJhUIIEatESihnCLvlfb/6Cg46yPhea0kizEaSVCiEEMmSrlwHB9nJY7jrLvjZz+C994yfJRjIbRIQCCG6Lo8Hz8h+lN+gyJusKK/uZ69FbrpyHRIQmkA4ZuCY8DyGVqh+fit4PPzlL/CHP8C558LQoWk6aZFSEhAIIbomjwfP1IupOqYRb2/QCrxtjVQ9e0n0oCBduQ5xMksgnL1iNpVHVFLWrQSloWwT1MyH37zeyJ8vWsOtt8KFF8KcOdCtW7rfgUgFySEQQnRN5eWUj/Xi7R3+ULYUCLIrYiGkaQQVZfo3p3Mm/+ayPR7noaYLyJPbxqwnOQRCiMwQYb1+WjU00FBs8VAWFAiKhdX78TZ5KR/rJe82KJ8Engo4nRd5inN5aOsECQa6GPl1CyGSJ8p6/bQqLaW0yeKhHGtXbPV+FMqYLtEK79JqLjtmIP+sgHP5F3ll+6f4LEW6SUAghEieKOv10zp6UF1N9ZsFFLUGby5ShVnZrjhS1UGzQkgKZSw77MiD+Q/DW39i++pxuEeT8QmSIjkkIBBCJE+kHgHpHj1wuXDd8Ag175RQtgkjsa5bCTVjZ2Vdx8FoVQfNCiFpNLTnw3OPwkeXwvApcNxdxjRKBidIiuSRpEIhRPJE6iII6elsmIPi6Z5Y9n8/oeGRO+CT82GUG07436ivEdlJkgqFEOkXab1+ujob5qB4uifeetwd5G3dH066sTMYSHU/BZFZJCAQQiRPpPX6mVTtL1NXQtjUt0df0+1myYTbt8OWLXDpL85n1jNeysY8nbZ+CiKzSLkJIURyuVzm89HV1UbOQGDSYTqS2fy5DP7z8OcyQFbMo3vqPGxp3RK2vSCvIOxuv6UFzjkHduyA2lqoHPIbKof8JlWnKjKcjBAIIdIjU6r9RVsJkeHctW5a21vDtvfarVfQ3f62bXDGGbBoEVRWQn5+Ks9SZAMZIRBCpI/V6EEqZXkug1WewIaWDZ3fb9litC5++2147DGYMCFVZyeyiYwQCCEiy/L59agyKZchDtG6FnrqPOx93ALefKuNvhOuQR2RY78/4RgJCIQQ1tJdKyBZwUjgfrduhYKC4MdDchkiFf1JN7OiQ/7VAv76BM0n3ADjx7H+wPuD6hOkUyZ/pl2V1CEQQliLVEcg2bUCQpP9wLhQJ5pnYLbfwkLo2RM2bDBGBqqrO4/ReVHduev5RQVFGZWR76nz4K5109DUQGlxKdWjqzllHxcHX3oHG3/+P6CCn5/uWgPZ8JnmErt1CCQgEEJYy8szRgZCKQUdHbHvz+MxkvUaGsIuvGGSFYzEuN94iv6YMbtoJ+vi98MPMHo0fPJ5C1zxc+j/edDjCkXHbXH8/hzi1Gcq7LEbEEhSoRDCWmmp+cUznvn1WJf3JSvZL8b9xlP0J1ToHbG/tDDgeFCwdq0RDDQ0wJ5Vl/BjSDAA6W/e5MRnKpwnOQRCCGuRKg3GKtblfVZBR15eYjkFpaV4Kox2v4Ftf62OV9rNftEfK+5ad9DwOEB71RF9AAAgAElEQVTzzmbctc4ubfz2Wxg+HNasgZdegr9feYZlfoGZVM3rR0uEFOkhAYEQwpqTtQJiveM3C0YA2tsTSnD03DKGqrMw2v4q42vVWcb28Cd7qH52c3hHxFaofn6r7WOn6o74449h40Z45RU4/njzpkZW8/TRGiSZiTeAiJQIKdJHcgiEEKkRT05AYM5BXp4RDITKzzfyGaLlJPhPI5b5a985eyrAPRoaiqG0CaprwVWH7SRHu8eMN89g+3bo3t34fssWIz8yVrHO6yeaGJjKnIquTpIKLUhAIESaJLpqwCrBMVRJCUyfbrnPvCl5RuvfEKaJdnaOaSPJ0c7FM9JzAMuL5+efw8knw9SpMG5c5FONJKbPBUkMzCbS7VAIkXl69Nj1fUlJbNMPdhMZGxuNwOPqq01rGMQ0f23nmDaSHO0M3VvlGVy/8HrLofxPPjFyBrZvh4EDo59qJFafS57KM50SkMTA3CMBgRAi+fyjA42Nu7a1tMS2D6ucAjPNzfDgg6YFlcYMNMkVAPPtdo5pM1BxVbion1RPx20d1E+qDxset7qQNrY0mgYKNz02lxEjjHjn9zXzObM2sWRAs3l9gHbdbppTIImBuUcCAiFE8jnRQMjlMrryKBX9uRA+1O873oIvF5g+3XR7YFIlhB/bwe6MMV1IN+/Ld/d56N4dfl/zAlM+OT+mZEAzoaMY+Sq8+1HgyghJDMw9EhAIIZLPqZoCCxbYyyOIcB4xD3W7XEaOgNYwZ07SujNaXWBLepSEP7nnWnqfWMOSJXDv6uscW9IYOIrRoc0LF/k/p1hWMIjsIIWJhBDJ51SBI7sBhFLmgUNeHqWb2vH2NjkVO3foSezO6L+QhiYPAruSDRuOgd2aKBrwDff97/4ccEDy5vJLi0tNkwYDPydXhSvmAEBWF2QuGSEQQjgvtCnRmDHOFDiyCiBKSoLv3K+80rKGQXUt4XUFzIa6HWysFG29vv/xic9MBGDOuDmdeQb+O/G91o2HOS+z2yszeeiMXXfiyZrLT8aUQDy1DkTqSEAghHCWWYfE2bON+X+T4faYittYVU6cPt0Y1u/oML7OmBFcUCl/13y4qw5q5kPZJlCa8KFujwf69YMJExzp8hjtImjnIrnn9y42z3qCQwfuQf1bv2DCoF131Mmay0/GlECqKjaK+CStDoFSahZwBvCj1vpw37bJwOXAOt/T/qS1XuB77I/ApUA7cJ3W+mXf9lOB6UA+8A+t9Z2+7QcATwB9gWXARK11SNwfTuoQCJFkMRQgiqu4TSwNkvzsNmkyq5UQ5T1EE229frTHFyww6gv89KeweDH07x9+jGwZho+11oFwRtoLEymlTgC2Ao+FBARbtdZ3hzz3UOBx4ChgX2AxcLDv4S+Ak4A1wAfABVrrVUqpJ4FntNZPKKUeBFZorR+Idl4SEAiRZDF0SExZcRu7QYrV8/zi6PIY7SIY6fH2Wzs44wz4/nujHHGJSX5hNpFiRumR9sJEWuslwAabTz8beEJrvUNr/Q2wGiM4OApYrbX+2nf3/wRwtlJKAaOAp32vnw2c4+gbEELEx2qe32R7yorb2G3SFC1pMY4uj9Hm+K0e379nGUrBU09BbW32BwMgSxUzXTpyCK5RSq1USs1SSvXxbdsP+DbgOWt826y2lwCbtNZtIduFEOkWQ4fElBW3sdukKdIFP86aA9EugmaPF666iO5z32PzZuOwvU1WRWQjWaqY2VIdEDwAHAQMBr4D/s+33azSiI5juymlVJVSaqlSaum6deusniaEcEIMHRJTesforyfgTzw0yzuwqkwYa5nlwMNGuQiGPl7y+e/Z+dQs9uu9J3k5mPYdrWKjSCOtddL+AeXAx9EeA/4I/DHgsZeBo33/Xg7Y/kffPwWsB7r5tgc9L9K/oUOHaiFE5pi7cq4um1qm1WSly6aW6bkr56b3mHPnal1WpjVonZ9vfC0rM7Yn2UMPGYc76SStt21L+uFEFwEs1TaujyktTKSU2kdr/Z3vx7HAx77vXwD+qZT6O0ZS4UDgfd+Ff6BvRcF/gfOB32ittVLqNeBcjLyCSuD51L0TIYRT4iluk4jQlQ3+ZX7+c+kcBQhcbeBfdghJK0z0yCNwxRVGyYZ//WtXO2MhUiWZqwweB0YA/YAfgNt8Pw/GGN6vB67wBwhKKTdwCdAGTNJaL/RtHwNMw1h2OEtrXe3bfiC7lh1+BEzQWu+Idl6yykCIrs1WpnsMSyed8vXX8Ne/GiUVdtstKYcQXVTalx1mKgkIhOjabK2Fj2HpZKIWLoRTTiEn8wVEZkj7skMhhMhEtlY2xLB0Ml5aw5QpxhTB3LmO7VaIuElAIIRIi4glix3sIxDK1soGs9UGhYV49llP+Q2KvMmK8up+cdfg1xr+/GeYPBkuuihpaQlCxEQCAiFEypnW739yIp5ByugjcPHF0fsIxBk02FoLH7p0sqQEz8/aqBq1DW9v0Aq8bY1UPXtJzEGB1nDzzfC//wuXXw4zZwa1WkiamHpGZNC+RepIDoEQIuUsE/s2Qf00ixcFJvSZ9RwoKrJfKyCwH0Lfvsa2DRuseyOUl1M+1mvaNrlsE9Q/W2avpwKwahX8/OdGMHDPPbHlDvh7FnibvOSrfNp1O2XFZVF7F8TVM8LkuGa9EhLdt0g+SSq0IAGBEOlnmdinoWOKxYsCE/oSWQUQrYGRWWCRl0ferRptUhKt85yjBCSBF9W9m0/kr67KoK6F0ZhdeDtPOcoFOJEeAtEu+NKfIPNJUqEQImNZJvY1RXpRwGuseg54vdGnENxu62AAjMfcIe14S0stz61zu9nrfOYs/yeVF7fhXXI8Gs13RYu44t/BLZCjDbmbtQ7uPOUoLYQT6RkRrWVxyvpRiKSTgEAIkXKmiX2tUF1r8YLQPgKRsv0j5R1A9AZGZs+prqb6zQKKQhqsh52zyb7b2uDKS4to/7ASNh7Qud1/UTXNp5hfFRYURLvARno8kZ4R0S74KetHIZJOAgIhRMp1JvZtMobcyzZBzXxw1QU8qaTEuheCVc+BQFZ37HaWDoY+x+XCdcMj1LxTEvmcQ163cyf85jfQvOwcGP1HGPGXoMcbmhqi3oF37jrKBTbS44n0jIh2wZcOhrlDAgIhhCNizTR3rYT6e/PpmGIkEoYFA+vXWzciCl0FYMVsNCBaMGHW1dCXhOh6YwP1z5bR8eNV1NcUBZ9zyOva2+HXvzbaF/c56w44/s6wQ5UWl9oecje78PoV5BVEvAAn0mUw2gVfOhjmDkkqFEIkzDTxTBVS81pPXG+YZO+HJPZ5KsA9GhqKoXSzonrglbiummH/BGwkGQZlynfrS/VijHOLtMrA44Hrr4fGxuD9FhVBZSUsWGAEHRarEyZPNmKbviOsE/P8qwbCTt0kKc9T5+H6hdfT2BJ8PoX5hcw6e1bSLsKRVhmIzCerDCxIQCCE82wtIwzMwu/Xr/Mi66mAqjOhuXDX62JdtuZ54GrcXz5IQy9NaZMxr+/6atfx4loaF201gsWKhuZmIzb52c9CdmdxUY313CSrX8RKVhkIIVLGcti7OOAH/5y+xxN0x+0eHRwMgG8O/YXrbRUe8tR5qNo4G2+xsSzQ2xuqzlZ47q7svGOPOE9vVeAo2moEk+mIbdvgjDNg+HDYvHnX+ZVPK2fiMxMBmDNuTueF27+9R7celPQosTXkLln9IllS2v5YCJGbSotLTe9aw5bqNTSEJfoFBQ2B23c2gtcXOERoP2x6se+mce9YgP+Z1hdRL7gt2hxHW40QkkC4eTOcfjq88w7Mng29elm3Wn674W1mr5jdub2xpZGigiLmjJsTdVTE8rOWrH6RIBkhEEIkzPYywtLSsAtt1PX9fharBuzcMVtmym/NDx8F8B8n0mqEkATCTZvg5JPhP/+Bxx+HCROM7VYjEzUf1thaWWBGsvpFskhAIIRIWFimebcSal4uMM/CD7nQVtcSfX2/nz+Y8A3zewYp8jrM86ACgwDLi+jL7eZvqKHBejVCSUnYMsjqali2DJ5+2lhZ0Lkbi2ClXZsf186wv2T1i2SRKQMhhCNcFa7gi1J5QL+A0Cz8gGQ9Vx1QWIB7bC8a2jYYCXfPb8VV1xh+kNLSzmQ/z0HNVJ0J7Sa3NaF3zP7zCkvqq3EDJqsTSkt3navVewjwl7/AOefAsceG7MZieN/fhyDssDaH/cM+ayEcIKsMhBCp54kQLPgft2pe5HaD10v5JEybDeWrfGaPnW3vgplAk6QffoDf/Q7uuw/69LHYvcUKgsojKoNyCPzb5U5fJIOsMhBCZC6Xy1iyZ7fwUGC1Qt+0gVUyYofusH9RjXScCP77X2MlwXPPwWefRdi9xfD+jNNnyLC/yDgyQiCEyC6+IkRWIwTJXo/f0ACjRhkjBAsWwPHHJ+1QGUcKFGUnGSEQQuSm6mooKDBPRlSFtrLtYy2z7E9i/EYdyPADv2X9d60sWtT1ggE7TZhE9pKAQAiRdDFfgCNxuaBXL1x1RnOhoGZDr/WMesca84XNn2fg9ZJPGyXtP1DbMZJffuWxLmqUg+w2YRLZS6YMhBCxiZYQGPr0eMoGR5OXZ7Q5DqWUkZcQQb+/9gvrBQARphrKy/nW286+rCWfDjSgwFh+2NISV0JiNsqbkocm/DNXKDpui/yZi/SSKQMhhPMC7pbReldlvwh3xkm5s7QqGhSltbGnzmMaDIB1DYCPvT0ZxlL+gNGtsLO3YmOjdVGjHBStDbLIfhIQCCHsM6vvH+UimGjtfdPpBrOiQWZti0NPP0IQUro1L2zof/lyGJH3Bvm0cykzbZ1v1JLHWUoqJOY+CQiEEPZZXewiXAQTubO0nO8fRFzLBS2DEI1RtTBg1GPpXxYyahQU9dmNJd1P4RA+3/X8oiJjysD0jeXmHbNUSMx9kkMghLDPt+QP4OrToGaYUSkwX0PVUVcx4/QZYS9JJIcg4Va/IfkO5ZdvxdsWPmVQsg3W/23Xzy1058B8L93335PXXoPyt03yJiDuokZCpJLkEAghnOcbqr/6NHjgKGjPB5QRFDyw9AGufvHqsJckcmcZsUthtOx+k3yH6mc3U9QRXLG9qBWmvxT80h5s55/t57NkibF700JKcRY1imdlgn/aRE1RdLu9G2qKSny1hhAhZIRACBEbj4duX0ww7SGQr/Jpu7XNsUNZjhA0KeqnBvy/y+zOPGA0I5BnkMI9StNQbHRUrK6lswnTq4yknnIu4RHjAl9f79h7MQ4ee6lksxGWzpdKuWNhg4wQCCGSwjPIvKEQhHTxc2CNvlkiGxq2dtN4KgK2mSU2WuQ1uFZq6qdBxxSon7YrGHiZkzmdF5nGJFp7FMOYMZ0dFctv6kaeE3flcSRlmq3S6Hyp1AEQDpKAQAhhm/9u1Uq+yvc9MfbliWb80w0leXvQuQReQePuUHUmnUGBpwLKx3qDVyLEkNz3b07nLF7gp3xO7YCLKLzoNzB7Np5eXqrOBO8e7WhIvDpfHEmZ0VZj2F2tIUQ0EhAIIWyLdLcKUDXUFyzEeiccYTTBtRL22LgtoACAb3eF4B5tBANVZxp9DYJWItwyJnxpoolnOYdxPMOgA7byauNg+n+7zGhS0NyMe7RxnKDjxnBXHrZkcnhf8ydGCF6ircaQOgDCKRIQCCFsi3Q3etWwgFUGsdwJRxtNcLtp6GWe69RQjPVFe8cCqKkxRg8mQd5txtegqQalWN37SIb+pInFH5XQ13+9jtJR0c5duemSyZFb8AwtCH5ilPoJ1buNoahNmT4mdQCEkyQgEELYZnU3WlZcFrzkMFIlwdDRgOuvjzya0NBAaZPF7poiX7Q9g6DqbGWMHihjFME/1bBp/wro6OCmjX/ijU/6Uxy4H9/5Wx7Xxl25aYVG3Yp7bC/7KxM8Hlw3zqbmeU3ZJkBDvq9KcKTVGo72jhBdhgQEQgjbbFers6okOGZM+GhAo3kp4c7RhNJS886GbYrqS+ZS2rvM9OWlxaXGRblb8OhCcyFcv8/FHLjhAz75xNhWGDLC4D9/0+PavCu3XDLZtiF8CaMV39SLq85IgNRToO120I8YdRisggHpSijiIQGBEMI22zUFrNbo++bmbfGPMlRX4/qqKLizYZOiZt8rcVW4IgYpphflpVU0vjKLI4/djQMPtHqjxvm7NpcZx92ajyLyXXnY6TtR+z+OJETpSijiJXUIhBCpY9WlMFTo2vwoHRY9dR7ctW4amhooLS6lenQ1rgpXeB2D966BhffS49BaNnw4mu7dHX5/ARzp8mhRSyFSjQTpSihCSR0CIURyJFJfwCq3oKQk8ry6WaXAAK4KF/WT6um4rSNoKD1o9OCzs2DhveQf+jwPzPkxqcGA/5wqj6jsXIqZr/KpPKIytiJCcTRxkq6EIl4SEAgh7Eu0voDVBW76dPvz6jEInOLgJy/R56w7mDlnG5VDLnBk/5F46jzMXjG7s1hTu25n9orZsc3lx1EeWboSinjJlIEQwr44hrDDRBn+d5LWMGMG/PrX0L9/Ug5hKeHGTAmwmkIRXZNMGQghnBdHklsY3/C/Z8Ucoz7A6omJLY2zmMLQGv70J7jmGnjwwfh2nQjrxkzJryxoNYUiRCTdoj9FCCF8SkvNRwhiKBMM4Ql3/qVxQGwXr9BmQb4pDK3h98tcTJ0KV1wRsVVA0pQWl5qOEMhcvshUMkIghLAvjiQ3M44tjTMpkdzR3MK1V7cxdSpcey088IAxeJBqMpcvso0EBEKI6PzD8hMnQo8exqoAm0luZszunAEaNnmDEhSjVtwzmapoophXthzNjTcauYrKpOpv0H6r++EZ2S+hroxmbNdsECJDyJSBECKy0GH5xkZjVGDOnLiSAT11HhTKdK18aRPGsTDaLEedVujbt7PSYTt5aBR92MT7+59L8V9XmgYDV794NQ8ufbDz+N62RqqOARrBVeftPL4TiY6uCpcEACJryAiBEDku4br2sXYujLa7Wrd54RwN1bW79h11WsHjgc2bAWgjn4nMoZLZ6IJCev+/WyxHBgKDgc79+jonmr63ROouCJFFJCAQIoc5UtfeiZUFgS+zyLLXgKtu176jZum73bBzJzvpxgU8zuP8hgrqUL16Wt7dWwUjENIkyf/eEq27IEQWkYBAiBzmSPJepM6FcbDsmBjYWbC0NHrFvYYGHj2skOIDn+ZpzqPPCTewf8VdsGGD5bEjLfkrDTk+4PjoiBCZTAICIXKY1QXQKqnPlEMrCzp3Z5Z93+qbLgCj9WB1ddQsfc/wvlzW8RgtX58NY37LxlHTjNbGw/taHtsqyOicrvAbM8b46vDoiBCZTAICIXKY5QUQZX/aIFr53Bjn2IOz740OhjXzjekCTwWUX91K3uoJuGvdVB5RaZml7z4R2n9xP5x1KRw1A/DlApxofWyzIENpuPL9gOkKMLoyguOjI0JkMildLEQO89R5mPjMRNN581hL6JqWw11J8AoECO9UGElAKWRPBVSdaVzUO3dl0h1w61Z4+WU47+P4uvoFvY9NmurakGAAjMCnoyN8hUWs7y/TpbCMtEgfu6WLJSAQIsepKSbp9sTWDteyle+iHrhebwx/gd3eBgHtkMsngbe3ya4CApfNm43R/Pfegz1vOYG1BW9GfH5Udnoz5OpFM9eDHdFJehkIIQDjAmkmlhK6lsmJg02CAbA/xx4w9B6U5R+4K18exMaNcNJJRjDw+OPw1/FXJF4J0E5+RJTWy1lLEiZFCAkIhMhxTpTQtVwCaHERtz3HHnBBDsryD9xVcSmNjTB6NHz0ETz9NJx7rkOVAONoL5wzJGFShJBKhULkuM4kvATa4Vo26ikogaKW8GFnuysQ/Bdet5vqWi9VZyuau+2axvQHLgsXwqefwvPPw2mnBb+3hCsBulxdIwAI5VCjKpE7JIdACBGVZQ7BmTVGYqFDc+yhiYt3jKpmwiBjX/7dC4dIDkGXIUmFFiQgECI+pqsMklin/7//hbPPhnvvhaOPTtphurZcTZgUQRwNCJRSxwLLtdbblFITgCHAdK11DNVNMoMEBEJkPq8XRo2Cdetg4UI49tg4dhJwsfMM74v7RGho25CSYEaITOL0KoMHgGal1BHAzYAXeCyB8xNCCFNffw3DhxtNDBctSiAY8PUg8ByuqTqmEW9bY/z9HIToAuwGBG3aGEo4G2NkYDrQM3mnJYTIeSYVDtesMYKBLVvg1VfhF7+Ic98BS+rco4OLHUEc/RySLOGOlEI4wG5AsEUp9UdgAvCiUiofKEjeaQkhcppFF8G9X/0np58Or70GQ4YksP+ApXPR6hukmyMdKYVwgN2AYDywA7hUa/09sB/wt6SdlRAit4UUxfmEQ/muuRfdbv0TDz4IgwYluP+A5QiR6htkAkc6UgrhAFsBgdb6e63137XWb/p+btBaSw6BEDGQYeEAAXfwHzGY4bzBhTzmXFGcgIJH1bVGN8VAnYWZYmzMlAyWRZ8yZARDdB0RAwKl1Bal1GaTf1uUUptTdZJCZDsZFg7hu4P/gGGM4lWKaOYBrnKu0EBABULXx4qad0oo61YSXNHQ35gpZNoi1UGB1UhFpoxgiK5D6hAIkQLl08pNK/3F2nEwZ3g8vHPpTE7b8SwlNPIqoygvWgc1NXgGJVZV0TY7jY1iFce6/ohFn2RppHCA482NlFLHKaUu9n3fTyl1QCInKERXkqvDwlbTINGmR/RvXPx+/3ns1a2RNxhBeRlGMLD5baqenGg6kmK5z3iH/Z2s5e/xQL9+MGFCzCMOjvRkEMIBdgsT3QYMA36qtT5YKbUv8JTWOp4VwmklIwQiHXJxhMDqzrbyiEpmr5gd9Y73+++NBoL77uvfoYfyDyfiLQ7/f1JJjxJa2lrC99mnEteNs+Mrv+vUCIFZCeBE9ieEw5weIRgLnAVsA9Bar0XqEAhhmxMdB2ORigRGq+z4mg9rLLPmX37ZuIneuRP23jsgGABwu2noZX6D0tjSaL7PLx6Iv4WvndbHdpi1EQ4k3QNFlrAbELT6ChNpAKXU7sk7JSEyixMX11QOC6cqgdFquqNdt5tu974/iLPOgk8+ga1bzZ7gtVwiaHkOVu2X7VyEnWp9HOVYnuF9ZXWJyAp2A4InlVIPAb2VUpcDi4GHk3daQmQGJy+urgoX9ZPq6bitg/pJ9UmbI3ZiXbudIMgqCz5f5YdvXDUW5v2LQYOgthb69Ak9oAeUMl8iuBNKLG7ALQOIaKsV/HkHEycaP8+ZYwzrx9PYJ8KxPEMLqBq5RVaXiKxgtw7B3cDTwL+Ag4Fbtdb3JvPEhMgE2Vg0JtEERrtBkNU0yIjyEcE7/ORceOpJ9jr4WxYvhr59TQ7qdoPWuOqgZj6UbQKlja81tbszfaFJoNBq1BgIE23Y36JKYtzLDc2mHgBKSnCP7UWzDj7xTP/7EV2X7VUGQB3wJrDE970QOS/ei2ukO+xkz+8nsq7dU+eh8tlKW0GQ1TTI6g2rg3fa61v4yUsUnjyE4n9bvNeGBjwVUD4JJo4zNs15Buqngeu9ZvNAYT64Qv9PZGfY32zO327egRmzqYe5c2H9ehraNpi/3U3eXasirr467cWRhAD7qwwuA24FXgUUMBy4XWs9K7mn5zxZZSBiEc/qgEjryoGkrzmPd1272esCKRQdt3VEPX7elDw0Gn44HPb6eNfrNXT8zXwFgGdkP6qOaQxqQlTU6rvory0xWh9GM3euvSH/vDxjZCCUUsayBwdZ/v1sMoIdU3ZXSQhhk9OrDG4Cfq61vkhrXQkMBW5J5ASFyAbxrA6wmmaofLaS6xdeb+vuO5FRhHgTGM3OO5DdynmlxaXwwZXwQJ2RO+Df3kTQnXjge6wcsTG8I2Gh0amQ7dujH7SkxP4F1GrOv7TU8VLGpn8/VlMdfomMVgiRgG42n7cG2BLw8xbgW+dPR4jM4r+IxlI5L1L2fWOL+Z1u4GtC79T9c/iB52PnvGMdcYg0DRIYBHnqPBE/j+O/exrvi8Pg4Bfg4BeN1wdeBBsawt5jO+YjlQ3FwLZtkU+8qAimT7f3JsGY8w+tG1BUBGPGBG/35xZA3HfrYX8/mzTVtSZTHaFkqaJIg4hTBkqp3/m+HQxUAM9jLD08G3hfa31l0s/QYTJlIJLNapg4ksApiHQVMbI6br7KZ/bY2bgqXFGnI/72N7j5ZjhydAM/DPop3/baTmkTwRfBsjLKJ2HrM4o4tO7bV2B54GjBSmdpYa8X8vOhvX3XPvzbzY7hVGEhq2JIyTym6PKcmjLo6fv3FfAcdIbxzwPfJXSGQuQos2HiSEKnIJJV5jjaNITV9Ig/GIDIqy4++sgIBs4/H95eWIp36D/o+FuRkRjoDwZ8KwDsvJeoQ+tKBS0VjLo6InB1ARjBgH9FgstlfVfu9TqX8Ge1IiFQPMWRhHCANDcSIgn82fpmRXpKepSwR+EelnexyRghsEoYLOlRwvTTpnce33+H7W3ykq/yadftlBWXUb3bGFx3LSDvIi9ahe/fn3D4yiswahR0809GWjT7sRyNII8O3RE+qmAm5C466ucWqVRxdTVUVhpBQiROJPyFfiZjxsCCBTE1RBIiFnZHCOyuMugP3AwcBnT3b9daj0rkJNNBAgKRKk5m+ye6EiHSNEbovk2PvxNqXjCS/Ly9A16sgdemsFfFJ3x/3zzb5xP1PUYbWi8shFmzgi6cnasbQnSujrBaXQDGhT5S+eFAMpwvsozTqww8wGfAAcAUoB74IO6zE6ILiDfbPxlljiMN0YeucjCdFigwgoGgSoIaeGkqLLmVIVsmx3Q+Ud9jpKH1kpKwYABs1F+wWl2Qn28/GABJ+BM5y+4IwYda66FKqZVa60G+bW9orYcn/QwdJiMEoiuyk+iobzP+X2B5p62hYwp4KuBPoxQNb98HS6/mFNdnLJxzCMpkKiEqiykFs8c8t4zBvWOB5VRL1FEHs66EsYwM+KGNtP0AACAASURBVMkIgcgyTo8Q7PR9/U4pdbpS6ufAgLjPTgiRUtESHRWqM/nO8k7b1zfggjrFydMfgqVXc/PNJBYMRCoh7HIZF96ODjzzq6naODtiOeWoow5WzYzKyuyfsyT8iRxmd4TgDIyyxfsD9wK9gMla6/nJPT3nyQiB6Ko8dR6uX3i9ZS0Ef/JdpBwCVx20k8dF+XM54MzDmfJMRXzBAERO8gu5A0/qUkyzkQMz+fkwe7Yk/Ims4+gIgdb631rrJq31x1rrkVrrocBBCZ+lECKpApcaumvdTD/NuoCPP8/A9E57v6v4ddNB/MBe5Jftz+xHOrj92QSCAbCeizfZ3mAx3ZHoUkwgfOTASkeHBAMip8XS3CjU76I/RYiuJ9nNi2I5D7N1+SU9SkyfHzhVENqq+bxLZ3DBkas5fuD3bPuknryJDlwYI5UQDnojHkqbzC/UdsspWzJrg2w1hRCtpbIQWS6RgCCRewMhcpLd1sGpYFVECIipP8OOHXDuufCvfxmN+XZ/LvZ6/6ZBktlKArM5ereb6sU6vP1xm4rYUyL6SVnkMIwZY++8hMgxiQQEEZMPlFKzlFI/KqU+NnnsRqWUVkr18/2slFL3KKVWK6VWKqWGBDy3Uin1pe9fZcD2oUqpOt9r7lEqocFLIRwRqZJfqlkNpze2NNKjWw9KepREXtbo8dBS+lPO6b6Q+fNhxkXvM6l/lERAE5ZB0iDMk/xCh+UbGszbHz+vE+sQadUGecECe+clRI6J1stgC+YXfgX00FpbNkdSSp0AbAUe01ofHrB9f+AfwCHAUK31eqXUGOBaYAzwC2C61voXSqm+wFJgmO88PvS9ZqNS6n3geuBdYAFwj9Z6YbQ3LEmFIpmiFsdJoWhLDSMWO/LdPV/X/P+4j2t4mMu5tOgJ6NHDvBVxhKV4CScExpB8aMlseePEiXgO17hHG02UOqsjfux8G2Qh0smRpEKtdU+tdS+Tfz0jBQO+1y4BNpg8NBWj6mHg/zXPxggctNb6XaC3Umof4BRgkdZ6g9Z6I7AIONX3WC+t9X+0EdE8BpwT7c0KkWxRi+OkULSlhhFHLnx3z7dyO//iV1zKLOPuubERTwWUT4K824yvngoiFutJuDeD2dSCUkaQYGfKwmJqwPOLIqrONCovamV8rToTPL8ocrQFshDZIpEpg5gppc4C/qu1XhHy0H4Et1Ne49sWafsak+1Wx61SSi1VSi1dt25dAu9AiMjGDBwT0/ZkClwtYMXsotzUBH/yXkErBfSjkbE81/mYpwLzi+jwvpbHSDhI8q8CKAlIhvSPbNqYssDtxnNQc3AQc1Az7mNaaC4MfmpzIbiP3hbTlIgQuSJlAYFSqghwA7eaPWyyTcex3ZTWukZrPUxrPax///52TlcI2wIT5mo+rDF9zoIvF6T4rAz+1QJWQUHoRXnjRjjpJPgbN/I+R4U93z0a84voiebH99R52Nq6NWx7pCRG/+uCkhA3vw0tLeZPbm6G66+3vKv39PKaBjHenubTAg29TPbvTn0OSFJ5Yk8MFbkvlSMEB2H0QlihlKrHqHS4TCm1N8Yd/v4Bzx0ArI2yfYDJdiFSKjRhzqy7ITi0Xj4BVq2NAy/K69fD6NGwYgX864a3Oa7oo7D9NBSb77+hLXx20P/ZhBZCKulRErE3g2kS4toH8RwUoXBQY6PlXb37lHzTICbf4hbCX5Ex+A3mUP+CaBUiRZeVsoBAa12ntd5Ta12utS7HuKgP0Vp/D7wAXOhbbfBLoElr/R3wMnCyUqqPUqoPcDLwsu+xLUqpX/pWF1wIPJ+q9yK6JrOlc2arCsykI4cgULSyvj/+aLQtXrUKnn8ezvr7CNOyvn0t3qrZ+7P6bPYo3GNXbwGTu1TTlRrdjOQ/2wLu6hv2MA/S2vNMll+2KaprTZ6cSzUIrFZX5NooiIhZxMTARCilHgdGAP2UUmuA27TWMy2evgBjhcFqoBm4GEBrvUEp9Rd2dVa8XWvtvxW5CngU6AEs9P0TIilCy/n6l87ZCQaiDY+niqvCZXlX/t13sGEDvPiiMUpgvMBl/PO1DfZUwJbdwl9bkFdg+v4iJhOGlgv236VGep3F6IQl3119aXGZ5SqH6tHVuGvduxom7TYG11ezMf435JNrNQhiqBApuhZbvQxyiSw7FPGwWjqXr/JNpwnyVT4dusO0K18m2bwZevnmzLdvh+7dTZ7kW/ZXPsmYfw9V0qOE9TevB+gcNWloaiBP5Zl+NmXdSqh+cgPuUTp4uV8dUFZmHMfsAt6kqJ8aw/+vfMsSo3ZBDBWpA2Miz80UTizjFFnF6W6HQnRpVnet7brddG5+9tjZnWV/MzUY8Hrh5z+Hu+82fjYNBqBz2Z/VHfqGFmPQzk4+RZEqZMx7G6k6Q4evVPAtX7TMdxh4pf03F3BXH7ULYqiALovU10cOBrJxLt5uhUjR5UhAIIQNVjkA/ouL7YtNhvjqKzjhBGOa4IQTojzZt+yvdFu+6cP+z8YqZyBf5e/6bF7ryYKDOsxXKowGSkutL+BXzbDuM1BSElRZ0HN3JeXr3J35HkBQbwZHfj+JzsWnK9Pfqg10po9siKSTKQMhbIh52DmDff65kSfQ0gKLFsGQIdFfA9E/g0hVGueMm2NMJWzyGs8wWTisNHQMnBv5wmTWqrioKOiClrLflS+3IozyVTqMNJ1g430I4RSZMhDCQTEPO2eorVuN1QStrfD66/aDAYj+GViNovTt0XfXVILCsi1aaUFJ9IuhjbvblPWTiNStMdp0gmT6iwwkIwRCdDFz5xqBwKGHOrtfqzvzHt16hNUiCFWkCqkZO8uRACtl/SQi3eW73ZET96KNLgjhIBkhEEJ0WrYMXnnF+H7CBOeDAbAeQfAnHYbRvs6F3UocCwYghf0kIo1WRFvaF2l0QYg0kYBAiCjMChJlk/ffN3IGrr8e2tqSeyx/qeTA5D3LhMzeZXRM1tS718cfDJgk5tmpyugYqxUJ0S74kukvMpAEBEJEYFpGd35V1gQFb78NJ54IffvCSy9Bt6SVIrOWtAu0xTy9ayXpz/eIdsGXTH+RgSSHQIgIrAoSlRWXUT+pPvUnFIM33oDTT4f99oPaWhgwIPprkiWwYJFjxZoyvcBONhYtEjnJbg6BBARCRJCyBLUkuPZaePVVWLwY9tkn8f0FXdS79aV6Mbje2JC+i50k5glhi92AIA0DiEJkj9LiUtMRgnQ3K4pk504oKIDp06GpCfr0SXyfYb0c2hqpOgZoBFfdrj4EKQ0KSkvNRwgkMU+IuEgOgRARpDRBzQEvvACHH26MUuflORMMgMXafn91QUjPGnpJzBPCURIQCBFBNhUkuv7vSzh77E6+2PYex3kGOZr4aKsDYaq75WVSYl66yhAL4SDJIRAiB/z2rreY8adfwn7vges06L7F0XK9lsmVm6B+mv+HDEnmSzUpQywynBQmEiJF0l2nYP58mPHHo2H/t2DCKdB9C+BsuV7TqZNWo22x8cOuoXrLzyPwLrpfP+NfLtxRSxlikSNkhECIBGRC06NNm6DPqffA6D9AYUvQY06uhrCzysDy8+hTievG2eEXzs4nZfEdtax2EBlOlh1akIBAOClddQo8dR5umPY66/aZS1m/vdjautW0X0Cq6yVYfh5b86m/ux0AT4WRjNhQDKVNxiiDq47IUw6ZvKbfiXoImfz+RNaTKQMhUsAy2c5iuxM8dR4u/sMK1s16GP5zA94mL5t3bKYwvzDoeelYDWH5eey+KxioOhO8vUEr42vVmcZ2y6TEaJ0DsZimSFWiX6KrHWy8PyFSQQICkVVC/8d/9YtXp3X+PmWNdHw8dR4m/u5jdi74K/zsaTjmbgB2duykZ2HPtK+GsPw8tuUDxshAc3Dcsmv5Yl6e+UUwyhy9aXnpZy/BM/Vi5y+yZkGG2WqHykrj/OwEI5KDIDKETBmIrGE2Px0q1fP3TuQQ2C3r66nzcPENX7Gz9lY4/J8w9kLIb+98POXVE02GuT2DiJhDkHdTM1qF70pp6JiCeS5BlDl6WysgOjcmsBLC7mqCWFcdSA6CSDKZMhA5x6w4TignM+vtSLROQSzNk/7w3FR2vnM1HDEbxk0MCgYgxdUTY20sdNUMqKnpHCkIVdrk+8bszjhK50BbNRI6NyYwlWP3Tj7WO35phSwyhIwQiKxh1VcgVCrulJ1q1mN1d1vSo4Q9CvegoamB/XsZ+7/w2YnojaVQ3AB5wZ9DqkdG4k2kMx1RaYWa+b7EQgi/M45yx52yEQK7d/Kx3vGbvT+l4MorYcaM+M5ViAAyQiByjt074GTfKTvZEtnq7raxpdHYv9Y0zJvExZMa6NujL/TxhgUD+So/9fkCVnfaUe7Ag0ZUtHHRDgoGIPzOOEpFQtMaCR3ddtVICDRmTJQ3FoHdO/lY7/hdLiPnQAXMpWgNs2dLYqFIKQkIREYLTCLc2rqVgryCiM9PRWa9aV3/KFMVVsV6IgYvHQpenAHvTaKtpTtaY9pXYfbY2akvpZzAMLerwkX9pHo6Bs6lvqYoOBiwys53uYw7+44O42vAXLzptM2S4uD9+j35ZGwrDwKTCLduhcKQjEiz841n1cGCBeGjCpJYKFJMAgKRsULvxBtbGlFKUdKjpPN//FcNuyrlmfWxLjU0G1GY+MxE1BRlHeR05MH8h2HpVXDsnXDK79iwvZHKIyrTvpIAiH7Rs7Pkz8FeBJ1Bxm0d1E+qNwommWlsjL7ywH/uSsHEibue39hofC0piXy+8byvOEdchHCS5BCIjJWqoj+x5gPEel5Wz/crzC+kZ2FPNrRsoLS41Cgy9PjfYPnFMHwKjJgMvtHklOcKRGJVTCcTavtb5TiYCcwrMDv3SM+PV+hnt3WrEXAk41iiy5McApH1UlH0J558gFhbIkc739b2VvYo3KPz7nb6adMpHPgmjPoTjJzcGQxA7KsoktpnwWoYP9Ys+2QUEDIbwbASeBdudu6Rnh8PsxUamzfbm44QIokkIBAZKxVFf+LJB4h1qaGd8/U2eWlthXffNfY/639Gwwn/z/S5dgMiJ5MfYxLL8LeTVfoCAwu320jUCxy2Lykxf11g3oOdi32iywHNgo6dO6Fnz8xo5Sy6LAkIRMaK9U48HvGOQoTNWUcYwjd7H2F2dueXJ61hxAjjmuSqcFFWXGb61GgBhn9UYMIzE2IOdhwRS8JhpNEEi5EDyzLFoYHFgw8aX/3TGdOnR0/2i3axd+Ku3Sro2LDBMnFSiFSQgEBkrESL/oQyu5CkYhQi8H2Y2tkdnniOj5YMYNq0XdekeAKiwFEBK8nss8D/b+/cw+Soqr39rhkSSICEZCIql5kgRpCPKAIfco7KLVwjBA54jmADkYuRoJLow1ExPgJ6chRvGBVBPwm3DKB4ALkEAhIQ8SAXFRJUkIAzIYpABgwIIQkz+/ujqpOenqrqqu7q6uru3/s8/aRn16V37Ybea6+91m9Bsij7sMmx6Cko8xz0XnxmsNfjx3NGGhbF+KjivaBysF9Q34vpgGms2nt7PQMnCAkRiQajoELRFoRJDM9890yuePSKTMsX2/klQQHrx8LVN0PfATDjY7ifXzqi32kEPJaSSQXEuNX7woL/OjthcHDk6Wd30r/VyPZAEaIRJ8UM0KtX5cGogMVmLv8sco+CCoUoISxWYPGTi1P1QsRhmKfg96dC//7wbyfTc+BIJZ0kWxNQefWfWQVEP+Cw99GrmDwXOlacFBzUGOZNCDAGYFPVxBHtQTLFI06K6RmJ0DyoibCAxc5OGQMiF8ggEG1BVKxA0km3VoZtBezzfTh9X8bufUMqE3XUVkfWugWxghrDcvZ7QuInwmohjOqCUdGiVQ13yYcZJENDMgZELpBBIFqS8niBiWMmBp6XaUEgnyO2L/DOXzzOdhs+gJnRs9sLqUzUvct7+ef6f45oHztqLIuOXVRfY6c0AHDSJJg0iXkLYwY1Bq3IQzwH8982KziuYsYCGDcuvH95SOFTESORc2QQiJYjaGX6yvpXRigCZuY+L2H1apg2DZb/ekcuff+9qXklis88sHa4uE3XmK7E1RcT6xaUR/gPDMDAQKgbP1ZQY4jnoDD7B+FbPEHCPkXy4JKvRtJYiAxRUKFoOeJUEKylQmG1PPecZww89RTceCMcdljye5QHGU6fMp3FTy4ODSRMEkAYFnhZ0aAICQycPBf6t4noU5rBe729nsxw0O9ZV5dnieWBegUsChFB3KBCGQSi5Qgrk5xFWeQwnn0WDjrImwduvtl7n5SgCbsSSZ65kiRzaMZDSLnf3qkw6yh4rUSAb+x6+NH/dlF453941fzSkjeOkioeNQouu0wTr2hblGUg2pYstAWSstVWsOOOcPvt1RkDEJwpUYkkzxwVeBkZIBiyB15Y7pU27vkHw0sd3zPgiQYlkTeu2PmIbYgNG1Q1UIgYyEMgWo6qXd91YOVKz2O95ZbeIrq05H1SwjwfYSR95igPARDuPXjT/OiCQGbBrvywc4eq8OJUKmZU7X2FaAHkIRBtS2FqgZnvnkmneSlqndbJzHfPrIsx0Lu8l0lfn4Sdb9j5xqSvT9oYiPfUU/D+93uS+hDfGAgL7Euy2q8mxTBKGTHMe9C/pp/JL8yj47OvMfnsTnqn4llApSWCkyw6it6G3l4vW8HMe02aFF3foFIxo1IvRj2KKdWbZuyzaDrkIRAtR1Yegt7lvZxy4ylsGNowrH1052j+613X8Z3ZM1i3Dn7xC9hjj9r7DlSMIaj1OcPiBMK8B4YN81oU/+4Z37MpxiBuKeJiDAHAqafC+vXDjwfFApQG6U2cCK+/Dq++GnzfvJRmTkoz9lnkCgUVhiCDoLXpXd7LzBtmMuhGKtqVZxkUI/SrzToIlQl+/p10XPVLthq1FVuefjR/H/uL2PdPGthX6zPEJchQKTcGyokMICynp2dTxH2UAVEqPxw0URa3J4rSx6X3hfB7x5U1rpZasgsa1WfRMsggCEEGQetSTRR+KUlW173Leznx+hNHHhgy+OHv4dU3scVpR/L6hN8nun8eMySKlBsjlWomgF9j4EdjvX2Tiy8OPql8fz8ka2HY+d3d8M9/RmsPBK2io/Zt6vVbWOsKP2w8FBchYqIYAtF2VBOFX0rc0sBFwyOQDgfHnUDHqQcNMwbi3j+PGRJFClMLzJ82n+7x3axcs3JjjEYUK8fjTYSLF4fKEY/IUqik3FesfhhlDEBw1kJnSJ/D2tMgqsRzGKUxA6qOKDJCBoFoGdIo6xvnHoGGxzPvhaXng4PRb32KoYlPVHX/akoeZ0V56mHQtkw53Wv8NytXxlfqmz8fRo8mFcrTEUOKJoW216MPldrLlR+D+iaFQ1EHZBCIliFsFd1pnXSN6arpHqWMmNT73wdX3QHLP8IE3sbCoxcOr2iY4P6FqYXMqi8mlSkO88AUPQXGcHf82PUwv1jAsbs7vJBRudu8UICFC71MhVopX0WHeSnC2tOgUg2D8gyCOXPCqyJGjZsQNaIYAtEy1BqhP6pjFJcdc1mywL+/7A9X3wLjVrH9J2ey6rwHKvYlay2EIKrpX2h8g4OhKYvofRfMu2kOKzcM0L3GMwYKy0knIj4ssK6ry1N96u8fqXcQ9LmNiNiP+kyI1nAoRTEDokoUQyDajqjVdfmxrjFdbNax2bDrLaZQwEa3/lMHQ+9i2KafMR+bzgXHnRWrL0FUWq1XVXQogqDV/rAYh4C899D4hjXArFkUlkHfvNUMTVlE3w09njHQ2blpv7yW3Pmw7YYFC7xIe+fgqqvieR/ieCnSJOozg+ILwlDMgKgz8hCITAnVw8+YSul9lehd3svcb/+K1Us+xg5nfoyj3rNv1el/lVbr9fA2RGYzvP2qwBVt7zdnMuulK4b3Y70vR7ycyimBYSv2uOl4WRQGyrr4UKWMiiLSHRA1oLTDEGQQNI48udFrSe977jl485u994ODcO0fa3uuSsZJrcZL4s/8DqF57703z2fewhPpHw+dQzDYAT3F7YHHSlzacXLn8ya404j+VNoKUVVEkQLaMhB1pRoXdkU3dYZUm9533XWw006e+iB4HvGo54ozTlFFheIcr4bIbIaIqPjCMm/yH7sBBjsB80oczzrKq264cVsgTmR9Nel49STt/sSRG660FTI05P0rY0BkgAwCkZjIyncR14QJ2aSRLpiUatL7rr4ajj8e9twT9tlnU3uUzn+ccapknNRDmyAyxiEqKn7ePOZNG17SGLy/5x3kvBV2b3gFRDo6khkNWZJmf8pTB/v7N41NKY2IaRAiBBkEIjFJV/qRQj6kK7oT13ORNOjviivgxBNhv/28EsbjxlXuf6d1xhqnSsZJvbQJClML9M3tY+jcIfrm9m169ii9gJUrPbGhADaKEM2Z46kIBjE4WNloaFTwXJr9SeJtKBTkDRC5QAaBSExSF3aUgmCaojtJPRehE2IZDzwAp5wCBx8Mt97qbe2WEjZhhwn3lI9TJeMkS20C7wMjVq3d3ZvEhsrY2D4wEK0iWJwY4woVZUWa/cmb90OIGCioUCQmaZBbWAAfwKJjF6U2sdUj+A48j++ll3oegi22CD4nKHti3l3z6tKfhtLbS++FpzDrsA3Dtg2GZRvEoZhTn3VUfyXS6o8KEokcoSyDEGQQ1E7SbIF6TdTlpF0Y6OKL4cADYdddq+tPnrIqUqW3l94fz2HeHgOsHM9wEaK4tPrEmLcMCtHWKMtA1I2kLuys9PnTDL776lfhzDPhe9+rvj+Zu/qzolCgcPdq+i50m0SIHvO3FuLIDbeDDr+CBUUz4pxrq9dee+3lRPYsWrbI9VzY4+w8cz0X9rhFyxalfo9Fyxa5sfPHOs5j42vs/LFu9i2zY3/20JBz553nHDj3kY84t2FD4m7mhjTGPPmHLnJu7FhvAIuvUaOc6+pyzsy5nh7vnLj36ulJfl0jyUOf89AHkSuAh12M+bHhE3TWLxkEzUnYZB9kFJROgrNvmR3rOuc8Y+Ccc7z/Kz76UefeeCOrp0ufuONVnw9PYUIKMizGjs335JaHPuehDyJ3xDUIFEMgmoJq4xCSXLduHRx2GOyyixc/EFaGPg6NlmjOKm4jdYpBfUEBeZDv2IM8BBLmoQ8id8SNIdis0glC5IFq1friXOccrF3rbW3fdpuXSRCzzlEg5cGExfRHIDOjoB7qhnUnKBCvnDyn7eUh1TAPfRBNi4IKRVMQFTAYJUZUKdBwaAjOOAMOPRRefx3GjKnNGIBkwk21VDGs5blzSZzKf3mu+JcHoaU89EE0LTIIRFMQlqkwfcr0SDGiqAyHwUE47TQv+Hu//WDzzdPpa5SUcekEfuatZyaWgAbPEJj09UmceP2JVT13YuJo8qdBnFXs9Onx7pVVn0vJg9BSHvogmhYZBCKSeq1gkxKWwrf4ycWRq/Gw6z78zgInnwyXXw7nnef9XtbqGSgStgo3bNgEfsnDlyQu9lTcjhhYO1IJMM5zJ96yiKvJnwZxVrGLF1c+J8s+l5KHVMM89EE0LQoqFKHUIqyTtihPWJBetWJEZ53laQz893/DOeek05fS4+XPblioWmOSvocFC8a5tiqyDFI780wvmrMSlX6zsuhzNYqGeVNlFG2DlApDkEEQn1oi1dOMcj/z1jO55OFLhk2oReOiWnngJ5/0ShjPnp2oK7ENnXKjIWoST9L3KBnoStdWRUdH8ARclB5Ok7CJvJTOTnjjjehz6t3nalQIpVwoGoiUCkXN1BKpnlaUe+/y3hHGAGxyjyfZK3/9dfjhD725YsqU5MYAxA8YLC+c1DO+J/B+xvB9ikr7/FFBgfVQf8w0SC1ODMFgcMGoYUycmKw9KUkqGdZyjRAZI4NAhFJLpHpaUe7z7poXuiJeuWZl7L3y116DGTO8jIL770/UhRGfmaS9SJjhcsbeZyTa5w+6D0DXmK76yCJnGaQWx8joCTasMqWa1D6lA4omQDoEIpT50+YHusfjrEJrubaUqIm2aFwUphYiJ8JXX4WjjoJ77oGFC+Ff/zVRF0Z8ZpD7v5KhU+xfrWJFad0n/gf6981i73v+/GgdgriGyIsvJmtPSnd38NZGlEFTzTVCZE0cOcNWekm6OBm16OGnoaXfc2HPMPnd4svOs1j3e/ll597/fuesY9BtdfzHN17fdUFX1fUUGiYJ3A6Uyh53dW2qgVD6viiHHCaR3NMzXLq3+OrpSa+PSeWBJSksGgiqZSCDoBUImoDtPHOzb5m98XiU0XHPPc5tvsUG1/kfJ4wwKkZ9eVRdiiyJlAkrmDR6dPAEm9bkG1WToZp6DSo6JBpEXINAWQYi94Sl+UVF/H/4nQU28zfEdvzKXqwa+l3gvXOv7S/iZR8UKaYW1prip6wA0UIo7TAEGQStQ1hq4w6d72HbG37Hf/4nHH98dKpe6nn7In3C0giDSCu1UEWCRAuhtEPR8gQGHL7yZlZ990r++Efo6vKaogL+cq3tLzySBN6lFaSnrADRhsggEE3LxDFleeUvvxUuvwdb8zYWL4ZDDvGa50+bz+jO0SOuH9UxKv28fZE+QamPo0bB6LLvNM10yGr0FxpRP0GIFJFBIFIjzdoFcT7r5XUvb2p4fRxc/kt4ZXu++KP7OPDATYcKUwssPHohXWO6NrZ1jenismMuy6wcsaiBIH3+yy7zckjT0OwPmsiT6i80qn6CECmiGAKRCmnXLqjEiPgBB9z7Rca980HWXLQk9c8TLUpU8CDED0xUzIHIMYohEJkSV9I3LTbGDwzsDH+fCgbs/1+8su2ddfm8uBRLE9v5hp1vTPr6pLp6ShpO3tzkSfsTJSlcKHiT+dCQ92+U90ExB6IFkFKhSIW0ahfEpXt8N/0rtoArlsKYl2D2u6BjqKFBgr3LeznlxlPYMLRhY9vA2gFO/fmpAK23PVG+ui66yaExqXnV9CetiVxKhKIFkIdAZ6mQxgAAHzNJREFUpEJatQvickb39+Hye8F1wL//B3QM1ae4TwLm3TVvmDFQZP3g+khPSZaxF6mSt4I9Uf0J8xykVbwpy5oPQtQJGQQiFZJUHayVRx6Bb846km3GbsV2nypg2/4pVmGgehPlDQk7Voy96F/Tj8PRv6afWTfPag6jIG9u8rDPLXoKggL+0prIgwIfJWIkmgwFFYrUCFMUTJvjj4f//V9YuhTe/vbUb181YUJJEK6IGHZNUygo5i2QLqw/nZ3BZZPTUjUUIucoqLBJaVr3Md4eed/cPobOHaJvbl/qxkDRdl24EO67L1/GAHheklEdo0a0j+4cHeopyTr2IlXy5iYP60+QMQCeASBjQIiNyCDIEc3mPs7SePnVr+Dgg2HNGu83Po+xWoWpBS475rIRegcLj14YahxVE3uRG6Mxyk1eKdq/HtkJYf3p6Qk+f+JEaQcIUYK2DHJEM7mPs9QdWLoUjjoKdtzRe7/ddqnevqEkHces9R6qolJhoKwLB4V93pgxMDAw8nxpB4gWo+FbBma20MyeN7PHStq+YmbLzOwRM7vDzLbz283MvmtmK/zje5ZcM9PMnvRfM0va9zKz5f413zUzq9ezZEUzuY+z0h1YsgQ++EHYaSf45S9byxgAz6vwo6N+RM/4HgyrGByZtd5DVVTKPsg6OyHMc/Dii8HnSztAtCn13DK4HDi8rO0bzrl3Oef2AG4BvuS3HwFM8V+zgIsBzGwicC7wXmAf4Fwzm+Bfc7F/bvG68s9qOiq5j3PjKiYb4+WOO2DGDNhlF7j7bnjzm1O7da5IEnvRFEZjpeyDRmQnBIkMpZVyGIe8CTgJEUDdDALn3L3Ai2VtJeLzbAkba9IeDVzpPH4DbGNmbwUOA+50zr3onHsJuBM43D82zjl3v/P2PK4EjqnXs2RFVOpe3uILstAdeMc7PO/A0qXwpjeldtumJmu9h6qoNNFmORFHkVVQpOociCYh86BCM5tvZs8ABTZ5CLYHnik5bZXfFtW+KqA97DNnmdnDZvbwCy+8UPtD1Iko93HeXMX11B148EFvITd5Mlx/vRf7JTyy1HuomkoTbV6yE7LSDsibgJMQIWRuEDjn5jnndgR6gU/6zUH7/66K9rDP/JFzbm/n3N5vyvlSM8x9nDdXcdK977gsWgT/8i+wYEHw8TxtmzSCeo17qlSaaPMk4lO6lTB/vjdJp+3Wz5uAkxAh1DXLwMwmA7c453YPONYD3Oqc293Mfgjc45y7xj/2BHBA8eWc+7jf/kPgHv91t3NuV7/9hNLzoshzlkEUzZSBUC0LF8Lpp8MBB8DNN8OWWw4/3hQR9qI5qWfmQ94EnETb0fAsgyDMbErJnzOAx/33NwEn+9kG+wJrnHPPAkuAQ81sgh9MeCiwxD/2ipnt62cXnAz8PLsnyZ6mcBXXwCWXwGmnwSGHwC23jDQGoEki7EVzUk+3fl62SISoQN2qHZrZNXgr/ElmtgovW2C6me0CDAH9wBn+6YuB6cAK4DXgFADn3Itm9hXgIf+8LzvnioGKs/EyGcYAt/mvlqW4As5CGjhrVq2CT3/aCyD82c9giy2Cz8vbtoloIerp1i96GKSIKHKOhIlELnjwQdhjDxg9Ovycdtg2EQ1Cbn3RwuRyy6CdafdguCC++lW4/HLv/T77RBsD0PrbJsKnETn7cusLIYMgC/KmIdBonINzz4UvfAHuuWdT0aJKNEWEvaiNRuXs5ynzQYgGIYMgA/ISDJcHL4VzniHw5S/DKafApZd6v79xiaPql4fnbGtqWeHXK7gvTp+C1AyFaCNkEGRAHoLh8uClcA7OPhu+9jU44wz48Y+9UvVJiZrw8/CcbUlxwjWDk06qfoVfj+A+KQUKEQsZBBmQhtxsravePHgpzGDCBDjrLPjBD7zFWlIqTfh5eM62o3TChZF7QK+9BnPmxPMa1EPWWEqBQsRCBkEG1BoMl8aqt5FeiqEhePpp7/0Xvwjf+U6ybYJSKk349XhObUFUIGjCLWdgIN4KvR7BfVkoBap4kWgBZBBkQK3BcGmseuN4Keox8Q0Owqmnwt57w7PPem21FKquNOGHPefEMROrejZtQcSgmok1bIVej+C+OF6Haif03l6YNAlOPFFbEqLpkUGQEUlK3JaTxqq3kpeiHhPfG2/AySfDFVfA3LnwlrdUfauNVDJsgp5zVMcoXln/SuizRRlCcY2xtvYiVOvODzMkag3uK41n2Gwzb4Iut0JLvQ7VxhgUrxsYGHlMWxKiCZFB0ASkEYNQyUuR1AtRaQLcsAE+8hG4+mpPb+BLX6rNM1CkkmET9JzjNh/H+sH1gc9WyRCKY4y1vRchyM1f/LJ7eqCrK/i6IEOiVtd7eTzD4KD3r3PD+1Tqdag2xqDSVomKF4kmQ0qFTUAWRX06zu/ABRSMNIyhc4cS9+eCC+Dzn4dvf9uTJU6T3uW9iSSco56te3x3pPphHHVEKSjiTcRh0ry9vd6+0foSo2z0aK+aVenqP40CQ2GKg0WClAc7OoLFMMw8L0UYYddFfZYQDUBKhS1EGoI8lVb0SbwQcbwJc+fCjTembwzApu2Xq469CoCTrj8p0k0f9WyVPABxAkKDjIGo9pak6Oa/yvtOOOmk4Sv88okzaCJNIxug0qo86Hi1mQ1Rx6VyKJoQGQRNQi0xCHFc2kkyIcIm0f4XXuBTn/K2VDffHI4+OnYXE5PETR/1bJUMoTjGWKcFiymEtbcsYXvxc+Z4e0ilbNgwcqJPIxugmkm82syGoOvA2yKRyqFoQmQQtAFxVvRJvBCBk+i6Ldn8p7/gBz+A++5L/RFGkCTmIerZ4hhClYyxQTcY2Mew9pYlbIUfFHQHIyf6NDQI5s8PD1YxC57kq81sCLpu0SJYvVrGgGhKFEPQBiSJD4jDiBiC17em45rbYdW+XHVlBx/5SK09Hvl55TEDJ11/UmrPlDQmoRzFEPhU2lMvp3yPPSyGYOZMWLw4fungqOjVNvu9EwLixxBslkVnRGMJC5xLkqVQSmlmQv/f1zD62qUM/vXdXHttBx/6UE1dHUG58VHcGpg4ZiIDa0euPKt5psLUQk3BmfOnzQ8Msmy7Kozd3cEBfV1dsHbtyIm+fLVeGvVfnPynT/fyVovXFrchSs8vp6cnvJSxECIUbRk0AbXmuNejbHDRjb7qUy/xtjHv4X9+lr4xAOFbA0Cqz1TLGKsKo0/YXvyCBfFd8uUaBIsXJw80rBQTkBdVwbz0QwgfbRnknLRSDmt1i5fz0kswbpxXnOiNNzz9l3oQtd1x1bFXpfJMWaR1tg1R6YfVUG1KYFg/0khtTIO89EO0BXG3DGQQlJD2pJkGedyffvZZOPhg+MAH4JJL6vtZWTx/Hse4pUliNITpClSb4x92v64uLxgwK9J+LiEikA5BQvKqNpeH0sml/PWvcMAB3m/ZCSfU//OmT5meqL0a8jbGLU1SmeC0ix2FpTAODGTrss+i4JIQCZFB4JPXsrlpyBanRX8/7Lef5yFYsgT237/+n7n4ycWJ2qshT2Pc8iQVH0q72FFUCmOWtQfqUeZZiBqRQeCT11ViPQICq2Fw0Av4HhiAO++E970vm8/N4nvJyxi3BdWsjGstdlRKlGchy9V5Pco8C1EjMgh88rpKzEsEe2cnfO97sHQpvPe92X1uFt9LXsa4LWj0yrhQSFZsqZ79SLvMsxA1oqBCH0WaB/OnP8EDD8BHP9qYz9f30mLkIbo+D30QIkMUVJgQrRJH8thjXgDhF74AL7/cmD7oe2kx8rAyzkMfhMgh8hCIQH7/ezjkEK9I0dKlsMsuje5RMooppP1r+um0TgbdID3je3KRSipyQjH9sb/f2xMbHPSMg1q1E4TIGZIuFlXz0ENw6KGe8NDSpbDzzo3uUTLKtxmKRYaKqaSAjIJ2p3zbYNAvRBVHGlmIFkVbBmIE998PEybAvfc2nzEAwSmkRfKQSipyQFD6Y5FK0shCtCgyCMRG1q71/j3rLHj00eatBVMpJbHRqaQiB1RKMZRAkGhDZBAIAO66y/MG/Pa33t9bb93Y/tRCpZTERqeSihxQKcVQAkGiDZFBIFiyBI480kvP3mGHRvemdoKEhopIcEgAwcJARSQQJNoUGQRtzs03w4wZsOuucPfd8OY3V3efWks0p0lpqiJAp3UCKGUxjHYsw1uaeghelgEoBVG0NUo7bGPuv9+rTfCe93heggkTqruPxIOaGIn0CNHySJhIVGTvveHzn/dqE0QZA5VW/3ktDCVikLTYkEiXdvTOiNwiHYI25IYbvOJE224LX/lK9Lnlq/+gXP68FoYSMVAZ3sZR7p2RBoJoMPIQtBkLF8Jxx8F558U7P87qP6+FoUQMGl1sqJ2Rd0bkDBkEbcTFF8Npp3kqhN/6Vrxr4qz+VT64iVEZ3sYh74zIGTII2oQFC+DMM730whtvhDFj4l0XZ/WvAkRNjAr9NA55Z0TOUJZBG7B2Ley5J+y2G1xzDYweHf9aZRAIUSeU4SEyQlkGAoChIc8bcO+9cO21yYwBaO7Vf560EdoKRc7HQ94ZkTPkIWhRnINzz4U//xkWLYLNcp5PUixXvHLNSrrHd9dcpliejQbR6qveYsnklSs9175KJYsmQB6CNsY5OOccL6Vwq628xUeeKU7e/Wv6cbiNqY21rOiljdAgWjlyvmjs9Pd7/5MV0wTlAREtggyCJiCJ69s5+Mxn4IILYPZsb2FWVGXNK/WYvKWN0CBaOXI+jrGj7RLRxMggqECj96GTrp4/9zn4zndgzhy46CLvdynv1GPyljZCg2jlyPlKxk4cD4IMBpFjmmC6aBz1cGUnJenqecYM+OIX4cIL879VUKQek7e0ERpEK+saVDJ2KnkQtOUgco4MggjysA8dZ/U8OAi33+69f//7vdiBvBgDcTws9Zi8mzk7oqlp5cj5SsZOJQ/CnDmtG18hWoKcx543ljzsQ3eP76Z/TX9gO8CGDXDyyV5K4UMPeQWL8kKcOgil79PMMijeVwZAAygUWsMAKKf4TGFZBt3d3qq/nO5uzwswMBB831aIrxAtgdIOI5j8ncmBk3HP+B765val3LNgotLn/n2XAiecANdf7wURfvazmXQpNnkYPyEyIyrlct68YGMBPC9KX18mXRTtidIOUyAP+9Bhru8PvaPAhz7kGQMXXpg/YwDy4WERIjOitkuivACtEF8hWgJ5CCqQtmBOWtx0ExxzjJdJMHt2o3sTjDwEQvhMnhzsIejqgtWrM++OaC/kIQjht3/7baL0wcLUAn1z+xg6d4i+uX25MAbAyyZYtiy/xgDkw8Mi2pA8pvaFBSQuWNCY/ggRQNsZBEBD0gfT4JVX4IMfhPvu8/7efffG9qdSBoEi/UVF0p6885ra18rZF6JlaLstA9vOHB/33jeT63rNGjjiCHjwQe+37cMfbmx/goIdAbrGdLHgiAWa9EVlqql7UKmWQJhrXoF7oo3RlkEMag1uy0rF8KWX4JBDvLTCn/yk8cYABGs0AAysHWhK74toAEnrHsRZ/YcF7/X3528bQYicIQ9BlR6CrKrp/eMfcNBB8Ic/wHXXebEDeaDj/A4c4f/tNJP3RTSIjg5vYi/HzKvbXU6c1X/YOWbDP6uVKjAKUQF5CCpQa3BbViqGW28N73433HhjfowBqCwrrNRCUZGkdQ/iFE4KCt4rNwZACoFCBNCWBkEawW31zrF/9ln429+8SoWXXebFD+SJoAyCUlRESFQkad2DOAZEUPBemBdUCoFCDKPtDIK9ttsrlfTBelbTW7UK9t8fjj46/Les0RQzCLrGdI04ptRCEYukkfdxDYhCwdtCGBry/u3pCb5fK1RgFCJF2s4gSIt65dj393vGwHPPeSnKSYsUZVmuuTC1wOrPrmbRsYuUWijCiUotLJ+8o/b0q03da+UKjEKkSNsFFSZVKowibRXDp57yAghffhmWLIF99knenywCHYWITTWphfXqR1S6ohAtTNygQhkEOWL6dHjgAbjzTthzz+TXSypY5A7pAmSLDB8RQFyDQOWPc8Tll8Pzz1evQBhkDES1C1F34mQGiHQo98YUdRpARoGIhWIIGsyyZXD66bBhA2y7bW1yxJ3WmahdiLqTNLVQVE9SoSchypBB0EB+9zs48EC4/XYvzbBWBt1gonYh6o4C+rJD3hhRIzIIGsSDD8K0abDVVvDLX6azYOoZH5xeFdYuRN1RUZ/skDdG1IgMggbw61/DwQfDxIlw772w887p3FflhkUuSZJaKKpH3hhRIzIIGsDo0bDrrp5nIEwzpRpUbliINkbeGFEjSjvMkP7+TQaAc8NFh9LWNBBC5ASlAooGo+JGOeO222CXXby6BDDSGJh18yz61/TjcPSv6VcJYSFagTglm4XICTIIMuCmm+CYY2C33YIrFmZVOVEIkTFKBRRNhAyCOvOzn8Fxx8Eee8Bdd0HXyFpAda+cKIRoEEoFFE2EDII60tcHJ5zg1SS4806YMCH4vHpWTsyCLAsqCdFUTJwY3K5UQJFDZBDUkcmT4brrvEJF48aFn9fM6YKKfxAihN5er1JZOaNHKxVQ5BJlGdSBSy/1jIFp0+Jf06xZBiqoJEQIYYWdurpg9erMuyPaFxU3ahAXXQSf/KQXN5DEIChMLTSFAVCO4h+ECCEsTuDFF7PthxAx0ZZBilx4oWcMzJjRPllFzR7/IETdkJSwaDJkEKTEBRfAZz7jeQauuw4237zRPcqGZo5/EKKuSEpYNBkyCFLAOfjjH72Mgmuv9WKG2gXJJQsRgqSERZOhoMIacA7+8Q8vnXDQrzDc2ZnKrYUQQohUkHRxnXEOPvtZ2HtvL2C4s1PGgBBCiOZFBkEVOAdz58I3vwlHHBGuPdKMSGRIVKS310up6+jw/m2XCFohWhwZBAkZGoJPfAK++1349Kfhe9/zfhdbgTgiQzIY2hwV66keGVIi5yiGICFf/Sp84Qvwuc9570urFjY7lUSGigZDaSGmsaPGKoiwnQgT2+np8bS6RTBFQ6q00NHYsQoyFJmgGII68fGPe+JDSY2BWlfWWazMK4kMqSqjaGixnnqtsLNYuavqoWgCZBDEYMMG+MY3YN06L17gzDOTGwO16P1nVS8gTEyowzroOL8j0HsAUiVsKxoltlOvrYok963FcFDVQ9EEyCCowPr18OEPexkFt95a3T1qXVlntTIPEhkCGHSDOMK3lqRK2EY0SmynXivsuPet1SCRaqFoAupmEJjZQjN73sweK2n7hpk9bmbLzOwGM9um5Ng5ZrbCzJ4ws8NK2g/321aY2edL2ncyswfM7Ekz+4mZpS4H9PrrnvLgDTfAggVw7LHV3adWvf+s6gWUiwx1WuU8SqkSthmNEtup1wo77n1rNUikWiiagHp6CC4HDi9ruxPY3Tn3LuDPwDkAZrYbcDzwf/xrfmBmnWbWCVwEHAHsBpzgnwtwAXChc24K8BJwWpqdX7sWjjkGbrkFLr4Yzjqr+nvVqvefZb2AwtQCfXP7GDp3iCE3FHqeVAnbmELBCyAcGvL+zSIorl4r7Lj3rdUgkWqhaALqZhA45+4FXixru8M594b/52+AHfz3RwPXOufWOef+AqwA9vFfK5xzTzvn1gPXAkebmQEHAT/zr78COCbN/vf3w29/65UyPuOM2u5Vq95/o+oFhBkcPeN7GDp3iL65fTIGRDbUa4Ud975pGCSNMKSESEAjYwhOBW7z328PPFNybJXfFtbeBfyjxLgotgdiZrPM7GEze/iFF16I7NS6dd6/u+4KTz4Jp54a93HCqVXvv1H1AlS4SOSGeq2w495XLn/RBtRVh8DMJgO3OOd2L2ufB+wNHOucc2Z2EXC/c26Rf/xSYDGewXKYc+50v/0kPK/Bl/3z3+637wgsds5NrdSnKB2CNWs85cEPflDZQEV6l/cy7655rFyzku7x3cyfNl9eAdGe9PZ6PwwrV3qegfnztcoXTUFcHYLNsuhMKWY2EzgSmOY2WSOrgB1LTtsB+Jv/Pqh9NbCNmW3mewlKz6+KF1+Eww6DRx+Fs8+u5U6tRWFqQQaAEOBN/jIARAuT6ZaBmR0OfA6Y4ZwrDdm9CTjezDY3s52AKcCDwEPAFD+jYDRe4OFNviFxN/Ah//qZwM+r7dfq1TBtGixbBtdfX302gRBCCNGs1DPt8BrgfmAXM1tlZqcB3we2Bu40s0fM7BIA59wfgJ8CfwRuBz7hnBv0V/+fBJYAfwJ+6p8LnmHxGTNbgRdTcGk1/dywAQ4+GB5/HG66CY48supHFkIIIZoW1TIArrwSdtgBDjqoQZ0SQggh6kRuYwjywjPPwBNPeN6Bk09udG+EEEKIxtKWBkFfn+cNePVVePpp2HLLRvdICCGEaCxtZxCsWwf77QevvAJ33CFjQAghhIA2NAieeALGjYO774Y99mh0b4QQQoh80HYGgXNwzz2w++4VTxVCCCHahrbLMjCzF4D+mKdPwhNBEo1B499YNP6NR99BY2mV8e9xzr2p0kltZxAkwcwejpOqIeqDxr+xaPwbj76DxtJu49/I4kZCCCGEyAkyCIQQQgghg6ACP2p0B9ocjX9j0fg3Hn0HjaWtxl8xBEIIIYSQh0AIIYQQMgiEEEIIQRsYBGa20MyeN7PHStq+YWaPm9kyM7vBzLYpOXaOma0wsyfM7LCS9sP9thVm9vmS9p3M7AEze9LMfmJmo7N7uvwTMv5f8cf+ETO7w8y289vNzL7rj/EyM9uz5JqZ/hg/aWYzS9r3MrPl/jXfNTPL9gnzT9B3UHLsbDNzZjbJ/1vfQcqE/D9wnpn91f9/4BEzm15yTL9BKRL237+Zfcofzz+Y2ddL2tt3/J1zLf0C9gP2BB4raTsU2Mx/fwFwgf9+N+BRYHNgJ+ApoNN/PQW8DRjtn7Obf81PgeP995cAsxv9zHl6hYz/uJL3ZwGX+O+nA7cBBuwLPOC3TwSe9v+d4L+f4B97EPgX/5rbgCMa/cx5ewV9B377jsASPKGuSfoOsht/4Dzg7IBz9RuUzfgfCPwC2Nz/e1uNv2t9D4Fz7l7gxbK2O5xzb/h//gbYwX9/NHCtc26dc+4vwApgH/+1wjn3tHNuPXAtcLS/EjoI+Jl//RXAMXV9oCYjZPxfLvlzS6AY2Xo0cKXz+A2wjZm9FTgMuNM596Jz7iXgTuBw/9g459z9zvu/8Uo0/iMI+g58LgQ+y6bxB30HqRMx/kHoNyhlQsZ/NvA159w6/5zn/fa2Hv+WNwhicCreqgZge+CZkmOr/Law9i7gHyXGRbFdVMDM5pvZM0AB+JLfnHT8t/ffl7eLCpjZDOCvzrlHyw7pO8iOT/rbMgvNbILfpt+gbHgH8AHf1f9LM/u/fntbj39bGwRmNg94A+gtNgWc5qpoFxVwzs1zzu2IN/af9Js1/hlgZmOBeWwyxIYdDmjTd5A+FwM7A3sAzwLf8ts1/tmwGd7W177AfwI/9Vf7bT3+bWsQ+EFRRwIF39UJnnW3Y8lpOwB/i2hfjedS3aysXcTnauA4/33S8V/Fpu2e0nYRzc54+6OPmlkf3rj9zszegr6DTHDOPeecG3TODQH/D88lDfoNyopVwPX+1tiDwBBeIaO2Hv+2NAjM7HDgc8AM59xrJYduAo43s83NbCdgCl7A1EPAFD+adDRwPHCTb0jcDXzIv34m8POsnqNZMbMpJX/OAB73398EnOxHuu8LrHHOPYsX+HaomU3wXauHAkv8Y6+Y2b6+dX8yGv+KOOeWO+e2dc5Nds5Nxvux29M593f0HWSCH3tR5N+AYgS8foOy4Ua8vX/M7B14gYKraffxb3RUY71fwDV4LrkNeD98p+EFijwDPOK/Lik5fx5eNOkTlERL40Vf/9k/Nq+k/W14/8GsAK7Dj1rVK3L8/wfvB3AZcDOwvX+uARf5Y7wc2LvkPqf6Y7wCOKWkfW//Xk8B38dX39Qr+jsoO97HpiwDfQcZjD9wlT++y/AmobeWnK/foPqP/2hgkf/f7e+AgzT+TtLFQgghhGjTLQMhhBBCDEcGgRBCCCFkEAghhBBCBoEQQgghkEEghBBCCGQQCCGqwMwG/Sp9j5nZdb76YbX3OsDMbkmzf0KI5MggEEJUw1rn3B7Oud2B9cAZpQd9YSP9vgjRROh/WCFErfwKeLuZTTazP5nZD/DEXnY0s0PN7H4z+53vSdgKNtaWf9zM7gOOLd7IzPb3PQ+PmNnvzWzrxjySEO2HDAIhRNX4Gu5H4KnuAeyCVz75PcCrwBeBg51zewIPA58xsy3w9PuPAj4AvKXklmcDn3DO7eEfW5vJgwghZBAIIapijJk9gjfJrwQu9dv7nXO/8d/vC+wG/No/dybQA+wK/MU596TzpFIXldz318C3zewsYBu3qaysEKLObFb5FCGEGMFafxW/Ea+2Ea+WNgF3OudOKDtvD0JKxDrnvmZmt+Lpxv/GzA52zj0edK4QIl3kIRBC1IvfAO8zs7cDmNlYv7Lc48BOZrazf95Gg8HMdnZeNcYL8LwPu2bdaSHaFRkEQoi64Jx7AfgocI2ZLcMzEHZ1zr0OzAJu9YMK+0sum+unMj6KFz9wW8bdFqJtUbVDIYQQQshDIIQQQggZBEIIIYRABoEQQgghkEEghBBCCGQQCCGEEAIZBEIIIYRABoEQQgghgP8P1OCiecdA4roAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 576x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "price_untrained = [convert_label_value(y) for y in preds_on_untrained]\n",
    "price_trained = [convert_label_value(y) for y in preds_on_trained]\n",
    "price_test = [convert_label_value(y) for y in y_test]\n",
    "\n",
    "compare_predictions(price_untrained, price_trained, price_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
