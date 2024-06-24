# <h1>Hi, I'm Skin! <br/><a href="https://github.com/sirskin01/try01"> <a SQL Developer </a>, <a> Data Analyst</a>,</h1>

# [Project 1: SQL IMDB movie Analysis]
### Project Goal: This project is to analyze imdb movie database using sql queries. First the data was collected from kaggle, then loaded into the data warehouse for further analytics. The following questions were used to analyse this dataset

### Question 1:
![sql_1](https://github.com/sirskin01/try01/assets/144762826/99966dd4-787e-4456-b8f4-63a7a37d26ac)

### Question 2a:
![sql_2a](https://github.com/sirskin01/try01/assets/144762826/42025e2f-d191-474b-8f7c-46450678c9e4)


### Question 2b:
![sql_2b](https://github.com/sirskin01/try01/assets/144762826/4bcb7f59-a97a-4036-bb88-a98cbf24a59c)


### Question 2c: 
![sql_2c](https://github.com/sirskin01/try01/assets/144762826/340fd7b1-2123-480c-a074-bf8df09c3c79)


### Question 3a:
![sql_3a](https://github.com/sirskin01/try01/assets/144762826/1bbf26e2-3c40-49b4-a12f-a2d5cfcc2059)


### Question 3b:
![sql_3b](https://github.com/sirskin01/try01/assets/144762826/7705561f-ba99-4365-b4f7-ce46b10fdb7a)


### Question 4:
![sql_4](https://github.com/sirskin01/try01/assets/144762826/ddf6dad6-791d-4f26-b718-38c9642d44ab)


### Question 5:
![sql_5](https://github.com/sirskin01/try01/assets/144762826/0164552a-df67-4bd0-992f-678879e799d2)


### Question 6:
![sql_6](https://github.com/sirskin01/try01/assets/144762826/8ee98452-b820-4e3f-a4ce-a145b8b3e81f)



# [Project 2: Titanic Machine Learning Prediction with Decision Tree]

[Uploading{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ceea728c-8fc1-4d87-be8c-fcd608102d33",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/Users/skin001/Desktop\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/envs/env/lib/python3.11/site-packages/IPython/core/magics/osm.py:417: UserWarning: using dhist requires you to install the `pickleshare` library.\n",
      "  self.shell.db['dhist'] = compress_dhist(dhist)[-100:]\n"
     ]
    }
   ],
   "source": [
    "cd desktop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "bd198586-ee3f-4106-a047-5aae7abce6c1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[34mData\u001b[m\u001b[m/    log.csv\n"
     ]
    }
   ],
   "source": [
    "ls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "dde08224-5f47-48c7-b56c-727b4c33ef16",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test.csv     titanic.csv\n"
     ]
    }
   ],
   "source": [
    "ls ./Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "43080720-27b3-44d4-9721-a4b338baac2c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "%matplotlib inline "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "aee71e2b-e819-477c-9f67-e1b2e5ed5a55",
   "metadata": {
    "scrolled": true
   },
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titan = pd.read_csv('./Data/titanic.csv', sep=(','))\n",
    "titan.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "61b34c38-fe5c-4c4d-880a-3cf6526f9329",
   "metadata": {},
   "source": [
    "# Data Wrangling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a84215ea-a846-48d1-b814-0cf199fde27f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 12)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check the shape of the data\n",
    "titan.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b34dcffd-7cf4-47e9-8d65-0ae09e6fb54d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check the columns\n",
    "titan.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "39c9c35a-651e-4716-b171-7154254fdb60",
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the statistical Info of the data\n",
    "titan.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "059acd01-7d34-473c-b4b9-9b1db1ea4f64",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "# Check the Info of the data\n",
    "titan.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c5970a54-e1bb-465e-9b55-0c994aab4e6c",
   "metadata": {},
   "source": [
    "# Handling Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4aacbfa3-eb38-44d8-84bf-12c1ae1e0644",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check for Missing Values\n",
    "titan.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7f374c97-281b-4501-bf82-91ea0c675afd",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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5uXl4fHjxwq3chq+goiI3lNMpvUaMpkMHh4emDNnDgCgWbNmuHDhAmJiYjBo0CD5dR07dkRycjIePnyIH374AZ999hlOnDjx2nwXrxMZGYkZM2YolEmk5pAYWJapPiIifaELywf6ojztXxCDSgOIqlWrokGDBgpl9evXx6+//qpQVrFiRbi6usLV1RWtWrVCnTp1sHLlSoSFhcHBwQGZmZkK1xcUFCArKwsODg5K22U2TiovuN5MuobvSVIXlQYQbdq0weXLlxXKrly5gpo1a772eTKZDHl5eQAALy8vZGdnIykpCe7u7gCKsnbKZDKFo54vYzZOKi/4QUmkv/Rr/kHFAURISAhat26NOXPm4LPPPsNff/2FFStWYMWKFQCA3NxczJ49Gz169EDVqlXx8OFDLF++HPfu3cOnn34KoGjGwtfXF8HBwYiJicGLFy8wZswY9OvX761PYBAREema8rR/QQwqbaJs0aIFtmzZgg0bNqBRo0aYNWsWoqKiEBAQAAAwMDBASkoK+vTpgw8++AD+/v74559/cPjwYTRs2FBez7p161CvXj107twZfn5+aNu2rXwQQkREVB4JIv2vvGA2TiKi9xj3QBTRRByIcc59Ralnya2NotSjbsyFQUT0HtOFL299oW9LGBxAEBERiYDHOImozDhdTLqG70lSFw4giETED0oi/aVf8w8qnsIAitJ5DxgwAHZ2dqhQoQIaN26MU6dOKVzz999/o0ePHrCyskLFihXRokULpKamyh/v0KEDJBKJwm3EiBHv/mqIiIi0RN+ycao0A1Gczrtjx474888/UblyZVy9elUhnff169fRtm1bDBkyBDNmzIClpSUuXrwIU1NThbqCg4Mxc+ZM+X0zM7N3fClERESkKSoNIObNm4caNWpg9erV8rJatWopXPP111/Dz88P8+fPl5fVrl27RF1mZmalhq4mIiJxcFlNc/TtFIao6bxlMhn++OMPfPDBB/Dx8YG9vT08PT2xdevWEnWtW7cOlSpVQqNGjRAWFoZnz56984shIiLSFn0LJKXSDERxOu/Q0FD873//w8mTJzFu3DgYGxtj0KBByMzMxNOnTzF37lx88803mDdvHuLj4/Hxxx9j//79+PDDDwEA/fv3R82aNeHo6Ihz585h8uTJuHz5Mn777Tel7ebl5clzaRQTBIH5MIiI3oCnMEhd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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Heatmap to see all null values\n",
    "sns.heatmap(titan.isnull())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "6ca86ed2-2ce3-47b7-b408-3c25d197671c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop Cabin column, as it is irrelevant to the study\n",
    "titan.drop('Cabin', axis =1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "4958c6f7-7c13-4cd6-9c79-287588eee3fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Replace all null values in age column to its mean values\n",
    "titan['Age'].fillna(int(titan[\"Age\"].mean()),inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "6aa7f854-8c1c-445c-96b4-5e5c832fb147",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    S\n",
       "Name: Embarked, dtype: object"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Since Embarked is a ctegorical column, we use mode to fill its null values\n",
    "titan['Embarked'].mode()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "84e89031-8b80-498c-a143-a9b2c3e66d29",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now fill the null values with S\n",
    "titan['Embarked'].fillna('S', inplace =True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1ad415e1-21cd-4000-840b-368a8776319d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    0\n",
       "Survived       0\n",
       "Pclass         0\n",
       "Name           0\n",
       "Sex            0\n",
       "Age            0\n",
       "SibSp          0\n",
       "Parch          0\n",
       "Ticket         0\n",
       "Fare           0\n",
       "Embarked       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titan.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "63ce518c-acf8-40aa-b718-eced736327fb",
   "metadata": {},
   "source": [
    "# Data Analysis of how many people survived"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "3011a30f-5df5-4437-b55f-09c7bbb48f0e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived\n",
       "0    549\n",
       "1    342\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check those who survived and those who didnt\n",
    "titan['Survived'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "d34d593f-1d97-44a7-875e-a62500ca15e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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wwa/9WvWrFFUVJR++OEHSVJpaanuvvtuRUdHKyYmRsOGDdM333zjW19XV6fs7GxFR0erffv2evTRR8WvGwSCD7EDoMVERkYqMjJSa9eulcfjadR7PfbYY5ozZ46+/PJLjRw5Un379tWKFSv81qxYsUJ/+tOfznmt1WrVnXfeqZUrV56zfvjw4Wrbtq1qa2uVmpqqqKgobdu2Tdu3b1dkZKQGDx6smpoaSdJzzz2n/Px8LV26VB999JFOnjypNWvWNOq8ADQ9YgdAiwkNDVV+fr6WL1+u6Oho3XzzzfrrX/+qzz///JLfa/bs2br99tt1zTXXKCYmRhkZGfrXv/7l2//VV1+puLhYGRkZ5319RkaG1q5d6/sUx+12a/369b71q1atUn19vf75z3+qR48e6t69u5YtW6aSkhLfp0zz589XTk6ORowYoe7du2vx4sXcEwQEIWIHQItKT0/X0aNH9fbbb2vw4MHasmWL+vTpo/z8/Et6n759+/o9HzVqlL755hvt3LlT0o+f0vTp00fdunU77+vvuOMOtW7dWm+//bYk6c0335TValVKSook6bPPPtPhw4cVFRXl+0QqJiZG1dXV+vrrr1VZWaljx44pOTnZ956hoaHnzAUg8IgdAC2uTZs2uv322zVjxgzt2LFD9957r2bNmqWQkB//SfrpfS+1tbXnfY927dr5PXc4HBo4cKDv0tTKlSsv+KmO9OMNyyNHjvRbf8899/hudD59+rSSkpK0Z88ev8dXX3113ktjAIIXsQMg4BITE1VVVaWOHTtKko4dO+bb99OblX9JRkaGVq1apaKiIv33v//VqFGjfnH9xo0btW/fPn3wwQd+cdSnTx8dOnRIsbGxuvbaa/0eNptNNptNnTp10q5du3yvOXPmjIqLiy96XgAtg9gB0GJOnDihgQMH6rXXXtPnn3+uI0eOaPXq1Zo7d66GDRumiIgI9evXz3fjcWFhoaZPn37R7z9ixAidOnVKEyZM0IABAxQXF/ez6/v37y+Hw6GMjAx17drV75JURkaGOnTooGHDhmnbtm06cuSItmzZooceekjffvutJOnhhx/WnDlztHbtWh04cEAPPvhgs//QQgCXjtgB0GIiIyOVnJysefPmqX///rrhhhs0Y8YMjR8/Xi+++KIkaenSpTpz5oySkpI0efJkPfXUUxf9/lFRURo6dKg+++yzn72EdZbFYtHo0aPPu75t27baunWrEhISfDcgjxs3TtXV1bJarZKkRx55RGPGjFFmZqacTqeioqL0xz/+8RL+RgC0BIuXHwoBAAAMxic7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjPb/AKOCCK6SkJAQAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# check graphically those who survived vs those who did not survive\n",
    "sns.countplot(x='Survived', data = titan)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "116c504e-90f9-4f01-82d0-1211a26c074d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sex\n",
       "male      577\n",
       "female    314\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check the number of peole on board according to their genders\n",
    "titan['Sex'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2083aaf9-d7e5-4a00-96e2-9eb1aa940c32",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Sex', ylabel='count'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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N998U+fOndP48eMVGRnpN//777/XpEmT9Oqrr/pud+rUSSUlJSoqKlLnzp0DcRoA6hmxA8AYTz/9tLZu3apnn31WCxYs0I4dO66YExMTo5YtW0qSvvjiC911112qrq7Wu+++q8GDBzfwigE0BD6NBcAYycnJeu+999SjRw95PB6FhYXp1ltvvercqqoq/frXv9bjjz+uO+64Q08//bQOHDig+Pj4hl00gHrHKzsAGp1Tp05pyJAheuqpp9SlSxdFR0dr//79GjdunNLS0vTmm2/q3nvv1ZkzZzR37lzdfvvtOn78uDZu3Khf/vKX6tWrlyZPnqy3335bn3/+uaKionTffffJ4XBow4YNgT49AHWM2AHQ6FRWVmrmzJnasmWLjh49qurqaiUmJmrIkCF64YUXFBkZqTNnzujf/u3f9M477/g+an7vvfcqNzdXR48e1QMPPKDt27erb9++kqSvv/5aXbt21SuvvKLMzMwAnyGAukTsAAAAo/E9OwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIz2/wHc11fv7/aMbQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x = 'Sex', data = titan)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "4cd5db42-798f-4e5a-873e-b2ad02399e22",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Sex', ylabel='count'>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAGwCAYAAABPSaTdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAAApzklEQVR4nO3de3SU9Z3H8c/kSkIyiQGSkJIgiAWyXISgMJZFjIGASEEiAk0xKGI3BCykAo2LQdFtuFRBKUqLi+AWFhc50AoLQiNEhMglCiIKRRo22UNuYpNAMBeS2T9aZp1yEcIkM/nxfp0z5zDP88wz3yeeMe/zzDMTi91utwsAAMBQXu4eAAAAoCkROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwmo+7B/AEDQ0NOnPmjIKDg2WxWNw9DgAAuA52u13nzp1TVFSUvLyufv6G2JF05swZRUdHu3sMAADQCIWFherQocNV1xM7koKDgyX97YdltVrdPA0AALgelZWVio6OdvwevxpiR3K8dWW1WokdAABamO+7BIULlAEAgNGIHQAAYDRiBwAAGI1rdgAA8AANDQ2qra119xgexdfXV97e3je9H2IHAAA3q62tVX5+vhoaGtw9iscJDQ1VZGTkTX0PHrEDAIAb2e12FRUVydvbW9HR0df8crxbid1u14ULF1RaWipJat++faP3RewAAOBGFy9e1IULFxQVFaXAwEB3j+NRAgICJEmlpaUKDw9v9Fta5CMAAG5UX18vSfLz83PzJJ7pUgDW1dU1eh/EDgAAHoC/zXhlrvi5EDsAAMBoxA4AADAasQMAAC6ze/duWSwWlZeXN+nzTJo0SaNHj27S5yB2AADwYGVlZUpNTVVMTIz8/f0VGRmpxMRE7d27t0mf995771VRUZFCQkKa9HmaAx89BwDAgyUlJam2tlZr1qxR586dVVJSouzsbJ09e7ZR+7Pb7aqvr5ePz7UTwM/PT5GRkY16Dk/DmR0AADxUeXm59uzZo4ULF+r+++9Xx44ddc899ygjI0M//vGPdfr0aVksFh0+fNjpMRaLRbt375b0/29Hbdu2TXFxcfL399eqVatksVh0/Phxp+dbsmSJ7rjjDqfHlZeXq7KyUgEBAdq2bZvT9ps2bVJwcLAuXLggSSosLNSjjz6q0NBQhYWFadSoUTp9+rRj+/r6eqWnpys0NFRt2rTR7NmzZbfbXf+D+wec2WkmcbPedvcI+I68xY+5ewQA+F5BQUEKCgrS5s2bNWDAAPn7+zd6X7/85S/161//Wp07d9Ztt92mlStXau3atXrxxRcd26xdu1Y/+clPLnus1WrVQw89pHXr1mn48OFO248ePVqBgYGqq6tTYmKibDab9uzZIx8fH7300ksaNmyYPvvsM/n5+enll1/W6tWrtWrVKnXv3l0vv/yyNm3apPj4+EYf1/XgzA4AAB7Kx8dHq1ev1po1axQaGqof/ehHevbZZ/XZZ5/d8L7mz5+vIUOG6I477lBYWJiSk5P1n//5n471f/7zn5WXl6fk5OQrPj45OVmbN292nMWprKzU1q1bHdu/8847amho0JtvvqmePXuqe/fueuutt1RQUOA4y7R06VJlZGRozJgx6t69u1asWNEs1wQROwAAeLCkpCSdOXNGf/zjHzVs2DDt3r1bffv21erVq29oP/369XO6P378eJ0+fVoff/yxpL+dpenbt6+6det2xcc/+OCD8vX11R//+EdJ0saNG2W1WpWQkCBJOnLkiL766isFBwc7zkiFhYWpurpap06dUkVFhYqKitS/f3/HPn18fC6bqykQOwAAeLhWrVppyJAheu6557Rv3z5NmjRJ8+bNc/zR0O9e93K1P6vQunVrp/uRkZGKj4/XunXrJEnr1q276lkd6W8XLD/yyCNO248bN85xofP58+cVFxenw4cPO93+/Oc/X/GtseZE7AAA0MLExsaqqqpK7dq1kyQVFRU51n33YuXvk5ycrHfeeUe5ubn6y1/+ovHjx3/v9tu3b9exY8f0wQcfOMVR3759dfLkSYWHh6tLly5Ot5CQEIWEhKh9+/bav3+/4zEXL15UXl7edc/bWMQOAAAe6uzZs4qPj9fvf/97ffbZZ8rPz9eGDRu0aNEijRo1SgEBARowYIAWLFigL7/8Ujk5OZo7d+5173/MmDE6d+6cUlNTdf/99ysqKuqa2w8aNEiRkZFKTk5Wp06dnN6SSk5OVtu2bTVq1Cjt2bNH+fn52r17t55++mn97//+ryTp5z//uRYsWKDNmzfr+PHjmjp1apN/aaFE7AAA4LGCgoLUv39/LVmyRIMGDVKPHj303HPPacqUKfrNb34jSVq1apUuXryouLg4zZgxQy+99NJ17z84OFgjR47U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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Number of people who survived based on gender\n",
    "sns.countplot(x='Sex', hue='Survived', data=titan)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "48291ec7-1034-4861-a3b4-b5e3a75bd671",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: ylabel='Frequency'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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4wmvfv36qVWrViouLlZOTo4kqaysTOXl5XI4HJIkh8Ohp59+WtXV1YqLi5MkFRUVKSoqSqmpqb4MDwAAGMSnsDNx4kQtWrRI2dnZ6t27t0JCQnw6eF5enpYuXaq3335b7du399xjEx0drTZt2ig6Olpjx45Vfn6+YmNjFRUVpfvvv18Oh0MDBgyQJA0aNEipqam66667NHv2bDmdTk2bNk15eXnM3gAAAN/CzrJly/TWW29p6NChTTr4/PnzJUk33HCD1/aFCxdqzJgxkqTnn39eoaGhysnJUX19vbKysvTyyy97+oaFhWn16tW677775HA41LZtW+Xm5urJJ59sUm0AAMAMPt+g3K1btyYf3LKsf9indevWmjdvnubNm3fOPikpKXr//febXA8AADCPT+/ZeeCBBzR37tzzCisAAACB5NPMztatW7Vx40atWbNGV1xxhVq1auXVvmLFCr8UBwAA0FQ+hZ2YmBjddttt/q4FAADA73wKOwsXLvR3HQAAAM3Cp3t2JOnUqVNav369/vM//1PHjx+XJFVUVKi2ttZvxQEAADSVTzM733zzjQYPHqzy8nLV19fr5ptvVvv27fX73/9e9fX1WrBggb/rBAAA8IlPMzsTJ05U//799be//U1t2rTxbL/ttttUXFzst+IAAACayqeZnf/7v//Thx9+KJvN5rW9c+fO+stf/uKXwgAAAPzBp5kdt9utxsbGM7Z/++23at++fZOLAgAA8Befws6gQYP0wgsveNZDQkJUW1ur6dOnN/krJAAAAPzJp8tYzz33nLKyspSamqoTJ07ozjvv1P79+9WxY0e98cYb/q4RAADAZz6FnU6dOumzzz7TsmXLtGvXLtXW1mrs2LEaOXKk1w3LAAAAgeZT2JGk8PBwjRo1yp+1AAAA+J1PYee111772fbRo0f7VAwAAIC/+RR2Jk6c6LXe0NCg77//XjabTZGRkYQdAAAQNHx6Gutvf/ub11JbW6uysjINHDiQG5QBAEBQ8fmenZ/q3r27Zs2apVGjRunLL7/0126B89J56nuBLgEAEKR8/iLQswkPD1dFRYU/dwkAANAkPs3svPPOO17rlmWpsrJSf/zjH3Xttdf6pTAAAAB/8Cns3HrrrV7rISEhuvTSS3XTTTfpueee80ddAAAAfuFT2HG73f6uAwAAoFn49Z4dAACAYOPTzE5+fv55950zZ44vhwAAAPALn8LOjh07tGPHDjU0NKhHjx6SpH379iksLExXXXWVp19ISIh/qgQAAPCRT2HnlltuUfv27bV48WJdcsklkn540eDdd9+t6667Tg888IBfiwQAAPCVT/fsPPfccyosLPQEHUm65JJL9NRTT/E0FgAACCo+hR2Xy6UjR46csf3IkSM6fvx4k4sCAADwF5/Czm233aa7775bK1as0Lfffqtvv/1W//u//6uxY8dq+PDh/q4RAADAZz7ds7NgwQI9+OCDuvPOO9XQ0PDDjsLDNXbsWD377LN+LRAAAKApfAo7kZGRevnll/Xss8/q4MGDkqSuXbuqbdu2fi0OAACgqZr0UsHKykpVVlaqe/fuatu2rSzL8lddAAAAfuFT2Pnuu++UkZGhyy+/XEOHDlVlZaUkaezYsTx2DgAAgopPYWfy5Mlq1aqVysvLFRkZ6dl+++23a+3atX4rDgAAoKl8umfngw8+0Lp169SpUyev7d27d9c333zjl8IAAAD8waeZnbq6Oq8ZndOOHj2qiIiIJhcFAADgLz6Fneuuu06vvfaaZz0kJERut1uzZ8/WjTfe6LfiAAAAmsqny1izZ89WRkaGtm/frpMnT+rhhx/W3r17dfToUf35z3/2d40AAAA+82lmp3fv3tq3b58GDhyoYcOGqa6uTsOHD9eOHTvUtWtXf9cIAADgswue2WloaNDgwYO1YMECPfroo81REwAAgN9c8MxOq1attGvXLr8cfMuWLbrllluUmJiokJAQrVq1yqt9zJgxCgkJ8VoGDx7s1efo0aMaOXKkoqKiFBMTo7Fjx6q2ttYv9QEAgJbPp8tYo0aN0iuvvNLkg9fV1alv376aN2/eOfsMHjzY86bmyspKvfHGG17tI0eO1N69e1VUVKTVq1dry5YtGjduXJNrAwAAZvDpBuVTp07p1Vdf1fr169WvX78zvhNrzpw557WfIUOGaMiQIT/bJyIiQna7/axtX3zxhdauXatPPvlE/fv3lyS99NJLGjp0qP7whz8oMTHxvOoAAADmuqCw89VXX6lz587as2ePrrrqKknSvn37vPqEhIT4rzpJmzZtUlxcnC655BLddNNNeuqpp9ShQwdJUklJiWJiYjxBR5IyMzMVGhqqbdu26bbbbvNrLQAAoOW5oLDTvXt3VVZWauPGjZJ++HqIF198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xRozZoy1bds266uvvrLWrVtnHThwwNOnpf++qa6u9jqHRUVFliRr48aNlmW1/PP49NNPWx06dLBWr15tHTp0yFq+fLnVrl07a+7cuZ4+gTyHhB0/uuaaa6y8vDzPemNjo5WYmGgVFhYGsCr/+GnYcbvdlt1ut5599lnPtmPHjlkRERHWG2+8EYAKm666utqSZG3evNmyrB/G06pVK2v58uWePl988YUlySopKQlUmU12ySWXWP/zP/9j1PiOHz9ude/e3SoqKrL+5V/+xRN2TBjj9OnTrb59+561zYTxWZZlTZkyxRo4cOA52038fTNx4kSra9eultvtNuI8ZmdnW/fcc4/XtuHDh1sjR460LCvw55DLWH5y8uRJlZaWKjMz07MtNDRUmZmZKikpCWBlzePQoUNyOp1e442OjlZ6enqLHW9NTY0kKTY2VpJUWlqqhoYGrzH27NlTycnJLXKMjY2NWrZsmerq6uRwOIwaX15enrKzs73GIplzDvfv36/ExERddtllGjlypMrLyyWZM7533nlH/fv317/9278pLi5Ov/zlL/Xf//3fnnbTft+cPHlSr7/+uu655x6FhIQYcR5/9atfqbi4WPv27ZMkffbZZ9q6dauGDBkiKfDn0Ig3KAeDv/71r2psbDzjrc3x8fH68ssvA1RV83E6nZJ01vGebmtJ3G63Jk2apGuvvVa9e/eW9MMYbTbbGV8S29LGuHv3bjkcDp04cULt2rXTypUrlZqaqp07dxoxvmXLlunTTz/VJ598ckabCecwPT1dixYtUo8ePVRZWakZM2bouuuu0549e4wYnyR99dVXmj9/vvLz8/XII4/ok08+0e9+9zvZbDbl5uYa9/tm1apVOnbsmMaMGSPJjP9Pp06dKpfLpZ49eyosLEyNjY16+umnNXLkSEmB/5tB2AH0w8zAnj17tHXr1kCX4nc9evTQzp07VVNToz/96U/Kzc3V5s2bA12WXxw+fFgTJ05UUVGRWrduHehymsXpfxlLUlpamtLT05WSkqK33npLbdq0CWBl/uN2u9W/f38988wzkqRf/vKX2rNnjxYsWKDc3NwAV+d/r7zyioYMGaLExMRAl+I3b731lpYsWaKlS5fqiiuu0M6dOzVp0iQlJiYGxTnkMpafdOzYUWFhYWfcPV9VVSW73R6gqprP6TGZMN4JEyZo9erV2rhxozp16uTZbrfbdfLkSR07dsyrf0sbo81mU7du3dSvXz8VFhaqb9++mjt3rhHjKy0tVXV1ta666iqFh4crPDxcmzdv1osvvqjw8HDFx8e3+DH+VExMjC6//HIdOHDAiHMoSQkJCUpNTfXa1qtXL8/lOpN+33zzzTdav369/v3f/92zzYTz+NBDD2nq1KkaMWKE+vTpo7vuukuTJ09WYWGhpMCfQ8KOn9hsNvXr10/FxcWebW63W8XFxXI4HAGsrHl06dJFdrvda7wul0vbtm1rMeO1LEsTJkzQypUrtWHDBnXp0sWrvV+/fmrVqpXXGMvKylReXt5ixng2brdb9fX1RowvIyNDu3fv1s6dOz1L//79NXLkSM/PLX2MP1VbW6uDBw8qISHBiHMoSddee+0Zr33Yt2+fUlJSJJnx++a0hQsXKi4uTtnZ2Z5tJpzH77//XqGh3pEiLCxMbrdbUhCcw2a/BfqfyLJly6yIiAhr0aJF1ueff26NGzfOiomJsZxOZ6BL88nx48etHTt2WDt27LAkWXPmzLF27NhhffPNN5Zl/fAYYUxMjPX2229bu3btsoYNG9aiHgW97777rOjoaGvTpk1ej4R+//33nj7jx4+3kpOTrQ0bNljbt2+3HA6H5XA4Alj1hZk6daq1efNm69ChQ9auXbusqVOnWiEhIdYHH3xgWVbLH9/Z/PhpLMtq+WN84IEHrE2bNlmHDh2y/vznP1uZmZlWx44drerqasuyWv74LOuH1waEh4dbTz/9tLV//35ryZIlVmRkpPX66697+rT03zeW9cMTusnJydaUKVPOaGvp5zE3N9f6xS9+4Xn0fMWKFVbHjh2thx9+2NMnkOeQsONnL730kpWcnGzZbDbrmmuusT766KNAl+SzjRs3WpLOWHJzcy3L+uFRwscee8yKj4+3IiIirIyMDKusrCywRV+As41NkrVw4UJPn7///e/Wb3/7W+uSSy6xIiMjrdtuu82qrKwMXNEX6J577rFSUlIsm81mXXrppVZGRoYn6FhWyx/f2fw07LT0Md5+++1WQkKCZbPZrF/84hfW7bff7vX+mZY+vtPeffddq3fv3lZERITVs2dP67/+67+82lv67xvLsqx169ZZks5ad0s/jy6Xy5o4caKVnJxstW7d2rrsssusRx991Kqvr/f0CeQ5DLGsH73eEAAAwDDcswMAAIxG2AEAAEYj7AAAAKMRdgAAgNEIOwAAwGiEHQAAYDTCDgAAMBphBwAAGI2wAwAAjEbYAQAARiPsAAAAoxF2AACA0f4fSGqv0YIyIYUAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# The distribution of all ages on board (continous variable)\n",
    "titan[\"Age\"].plot.hist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "1eaeca4c-b26e-4422-be4c-6cb649e30f28",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: ylabel='Frequency'>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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smXLVFNT48vQAAAAAOCcfApSmZmZGjNmjG699Vb17dtXffv21a233qrRo0dr+fLllq6VlpamxMRExcfHe50vLCxUbW2t1/kBAwaoT58+ys/PlyTl5+dr8ODBcjgcZp+EhAR5PB7t37//J+9ZXV0tj8fjdQAAAABAY/m02UTnzp21fPlyPfHEE/riiy8kSRdeeKHCwsIsXWft2rXatWuXduzYcUab2+1WcHCwwsPDvc47HA653W6zzw9D1On2020/JSMjQ4888oilWgEAAADgtF/0Qt6SkhKVlJSof//+CgsLk2EYjf7u0aNHNXXqVL388svq1KnTLynDstmzZ6uiosI8jh492qz3BwAAANC6+RSkvvnmG40cOVIXX3yxxo4dq5KSEklSampqo7c+LywsVFlZma644goFBQUpKChIeXl5Wrp0qYKCguRwOFRTU6Py8nKv75WWlsrpdEqSnE7nGbv4nf58us/ZhISEyGazeR0AAAAA0Fg+Banp06erY8eOKioqUufOnc3zN998szZt2tSoa4wcOVL79u3Tnj17zGPYsGFKTk42/92xY0fl5uaa3zl48KCKiorkcrkkSS6XS/v27VNZWZnZJycnRzabTTExMb4MDQAAAADOyadnpN555x29/fbb6tWrl9f5/v3768iRI426RteuXTVo0CCvc2FhYerWrZt5PjU1Venp6YqIiJDNZtP9998vl8ul4cOHS5JGjRqlmJgY3X777Vq8eLHcbrfmzJmjtLQ0hYSE+DI0AAAAADgnn4JUZWWl10rUaSdOnGjSAPPUU08pMDBQSUlJqq6uVkJCgteugB06dFB2drYmT54sl8ulsLAwpaSkaMGCBU1WAwAAAAD8WIBhZYeI/zN27FjFxsZq4cKF6tq1q/bu3au+fftq4sSJamho0GuvvXY+aj1vPB6P7Ha7KioqWsTzUv1mveXvEtDKfLUo0d8lAAAAtAmNzQY+rUgtXrxYI0eO1M6dO1VTU6MHH3xQ+/fv14kTJ/Thhx/6XDQAAAAAtAY+bTYxaNAgffbZZxoxYoTGjRunyspKTZgwQbt379aFF17Y1DUCAAAAQItieUWqtrZWo0ePVmZmpv7rv/7rfNQEAAAAAC2a5RWpjh07au/eveejFgAAAABoFXz6ad9tt92mlStXNnUtAAAAANAq+LTZRF1dnV544QW9++67io2NVVhYmFf7kiVLmqQ4AAAAAGiJLAWpL7/8Uv369dPHH3+sK664QpL02WefefUJCAhouuoAAAAAoAWyFKT69++vkpISvffee5Kkm2++WUuXLpXD4TgvxQEAAABAS2TpGakfv7t348aNqqysbNKCAAAAAKCl82mzidN+HKwAAAAAoD2wFKQCAgLOeAaKZ6IAAAAAtDeWnpEyDEN33nmnQkJCJElVVVW69957z9i17/XXX2+6CgEAAACghbEUpFJSUrw+33bbbU1aDAAAAAC0BpaC1KpVq85XHQAAAADQavyizSYAAAAAoD0iSAEAAACARQQpAAAAALCIIAUAAAAAFhGkAAAAAMAighQAAAAAWESQAgAAAACLCFIAAAAAYBFBCgAAAAAsIkgBAAAAgEUEKQAAAACwiCAFAAAAABYRpAAAAADAIoIUAAAAAFhEkAIAAAAAiwhSAAAAAGARQQoAAAAALCJIAQAAAIBFBCkAAAAAsMivQWrFihUaMmSIbDabbDabXC6XNm7caLZXVVUpLS1N3bp1U5cuXZSUlKTS0lKvaxQVFSkxMVGdO3dWZGSkZs6cqbq6uuYeCgAAAIB2xK9BqlevXlq0aJEKCwu1c+dOXXfddRo3bpz2798vSZo+fbo2bNigdevWKS8vT8XFxZowYYL5/fr6eiUmJqqmpkZbt27V6tWrlZWVpXnz5vlrSAAAAADagQDDMAx/F/FDEREReuKJJ3TTTTepR48eWrNmjW666SZJ0qeffqqBAwcqPz9fw4cP18aNG3X99deruLhYDodDkpSZmamHHnpIx48fV3BwcKPu6fF4ZLfbVVFRIZvNdt7G1lj9Zr3l7xLQyny1KNHfJQAAALQJjc0GLeYZqfr6eq1du1aVlZVyuVwqLCxUbW2t4uPjzT4DBgxQnz59lJ+fL0nKz8/X4MGDzRAlSQkJCfJ4POaq1tlUV1fL4/F4HQAAAADQWH4PUvv27VOXLl0UEhKie++9V+vXr1dMTIzcbreCg4MVHh7u1d/hcMjtdkuS3G63V4g63X667adkZGTIbrebR+/evZt2UAAAAADaNL8HqUsuuUR79uxRQUGBJk+erJSUFB04cOC83nP27NmqqKgwj6NHj57X+wEAAABoW4L8XUBwcLAuuugiSVJsbKx27NihZ555RjfffLNqampUXl7utSpVWloqp9MpSXI6ndq+fbvX9U7v6ne6z9mEhIQoJCSkiUcCAAAAoL3w+4rUjzU0NKi6ulqxsbHq2LGjcnNzzbaDBw+qqKhILpdLkuRyubRv3z6VlZWZfXJycmSz2RQTE9PstQMAAABoH/y6IjV79myNGTNGffr00cmTJ7VmzRq9//77evvtt2W325Wamqr09HRFRETIZrPp/vvvl8vl0vDhwyVJo0aNUkxMjG6//XYtXrxYbrdbc+bMUVpaGitOAAAAAM4bvwapsrIy3XHHHSopKZHdbteQIUP09ttv67e//a0k6amnnlJgYKCSkpJUXV2thIQELV++3Px+hw4dlJ2drcmTJ8vlciksLEwpKSlasGCBv4YEAAAAoB1oce+R8gfeI4XWjvdIAQAANI1W9x4pAAAAAGgtCFIAAAAAYBFBCgAAAAAsIkgBAAAAgEUEKQAAAACwiCAFAAAAABYRpAAAAADAIoIUAAAAAFhEkAIAAAAAiwhSAAAAAGARQQoAAAAALCJIAQAAAIBFBCkAAAAAsIggBQAAAAAWEaQAAAAAwCKCFAAAAABYRJACAAAAAIsIUgAAAABgEUEKAAAAACwiSAEAAACARQQpAAAAALCIIAUAAAAAFhGkAAAAAMAighQAAAAAWESQAgAAAACLCFIAAAAAYBFBCgAAAAAsIkgBAAAAgEUEKQAAAACwiCAFAAAAABYRpAAAAADAIoIUAAAAAFhEkAIAAAAAi/wapDIyMnTllVeqa9euioyM1Pjx43Xw4EGvPlVVVUpLS1O3bt3UpUsXJSUlqbS01KtPUVGREhMT1blzZ0VGRmrmzJmqq6trzqEAAAAAaEf8GqTy8vKUlpambdu2KScnR7W1tRo1apQqKyvNPtOnT9eGDRu0bt065eXlqbi4WBMmTDDb6+vrlZiYqJqaGm3dulWrV69WVlaW5s2b548hAQAAAGgHAgzDMPxdxGnHjx9XZGSk8vLy9Otf/1oVFRXq0aOH1qxZo5tuukmS9Omnn2rgwIHKz8/X8OHDtXHjRl1//fUqLi6Ww+GQJGVmZuqhhx7S8ePHFRwcfM77ejwe2e12VVRUyGazndcxNka/WW/5uwS0Ml8tSvR3CQAAAG1CY7NBi3pGqqKiQpIUEREhSSosLFRtba3i4+PNPgMGDFCfPn2Un58vScrPz9fgwYPNECVJCQkJ8ng82r9//1nvU11dLY/H43UAAAAAQGO1mCDV0NCgadOm6ZprrtGgQYMkSW63W8HBwQoPD/fq63A45Ha7zT4/DFGn20+3nU1GRobsdrt59O7du4lHAwAAAKAtazFBKi0tTR9//LHWrl173u81e/ZsVVRUmMfRo0fP+z0BAAAAtB1B/i5AkqZMmaLs7Gxt2bJFvXr1Ms87nU7V1NSovLzca1WqtLRUTqfT7LN9+3av653e1e90nx8LCQlRSEhIE48CAAAAQHvh1xUpwzA0ZcoUrV+/Xps3b1Z0dLRXe2xsrDp27Kjc3Fzz3MGDB1VUVCSXyyVJcrlc2rdvn8rKysw+OTk5stlsiomJaZ6BAAAAAGhX/LoilZaWpjVr1ujNN99U165dzWea7Ha7QkNDZbfblZqaqvT0dEVERMhms+n++++Xy+XS8OHDJUmjRo1STEyMbr/9di1evFhut1tz5sxRWloaq04AAAAAzgu/BqkVK1ZIkv7jP/7D6/yqVat05513SpKeeuopBQYGKikpSdXV1UpISNDy5cvNvh06dFB2drYmT54sl8ulsLAwpaSkaMGCBc01DAAAAADtTIt6j5S/8B4ptHa8RwoAAKBptMr3SAEAAABAa0CQAgAAAACLCFIAAAAAYBFBCgAAAAAsIkgBAAAAgEUEKQAAAACwiCAFAAAAABYRpAAAAADAIoIUAAAAAFhEkAIAAAAAiwhSAAAAAGARQQoAAAAALCJIAQAAAIBFBCkAAAAAsIggBQAAAAAWEaQAAAAAwCKCFAAAAABYRJACAAAAAIsIUgAAAABgEUEKAAAAACwiSAEAAACARQQpAAAAALCIIAUAAAAAFhGkAAAAAMAighQAAAAAWESQAgAAAACLCFIAAAAAYBFBCgAAAAAsIkgBAAAAgEUEKQAAAACwiCAFAAAAABYRpAAAAADAIoIUAAAAAFjk1yC1ZcsW3XDDDYqKilJAQIDeeOMNr3bDMDRv3jz17NlToaGhio+P16FDh7z6nDhxQsnJybLZbAoPD1dqaqpOnTrVjKMAAAAA0N74NUhVVlZq6NChWrZs2VnbFy9erKVLlyozM1MFBQUKCwtTQkKCqqqqzD7Jycnav3+/cnJylJ2drS1btmjSpEnNNQQAAAAA7VCQP28+ZswYjRkz5qxthmHo6aef1pw5czRu3DhJ0osvviiHw6E33nhDEydO1CeffKJNmzZpx44dGjZsmCTp2Wef1dixY/WnP/1JUVFRzTYWAAAAAO1Hi31G6vDhw3K73YqPjzfP2e12xcXFKT8/X5KUn5+v8PBwM0RJUnx8vAIDA1VQUPCT166urpbH4/E6AAAAAKCxWmyQcrvdkiSHw+F13uFwmG1ut1uRkZFe7UFBQYqIiDD7nE1GRobsdrt59O7du4mrBwAAANCWtdggdT7Nnj1bFRUV5nH06FF/lwQAAACgFWmxQcrpdEqSSktLvc6XlpaabU6nU2VlZV7tdXV1OnHihNnnbEJCQmSz2bwOAAAAAGisFhukoqOj5XQ6lZuba57zeDwqKCiQy+WSJLlcLpWXl6uwsNDss3nzZjU0NCguLq7ZawYAAADQPvh1175Tp07p888/Nz8fPnxYe/bsUUREhPr06aNp06bp0UcfVf/+/RUdHa25c+cqKipK48ePlyQNHDhQo0eP1j333KPMzEzV1tZqypQpmjhxIjv2AQAAADhv/Bqkdu7cqd/85jfm5/T0dElSSkqKsrKy9OCDD6qyslKTJk1SeXm5RowYoU2bNqlTp07md15++WVNmTJFI0eOVGBgoJKSkrR06dJmHwsAAACA9iPAMAzD30X4m8fjkd1uV0VFRYt4XqrfrLf8XQJama8WJfq7BAAAgDahsdmgxT4jBQAAAAAtFUEKAAAAACwiSAEAAACARQQpAAAAALCIIAUAAAAAFhGkAAAAAMAighQAAAAAWESQAgAAAACLCFIAAAAAYBFBCgAAAAAsIkgBAAAAgEUEKQAAAACwiCAFAAAAABYRpAAAAADAIoIUAAAAAFhEkAIAAAAAiwhSAAAAAGARQQoAAAAALCJIAQAAAIBFBCkAAAAAsIggBQAAAAAWEaQAAAAAwKIgfxcA4JfrN+stf5eAVuarRYn+LgEAgFaNFSkAAAAAsIggBQAAAAAWEaQAAAAAwCKCFAAAAABYRJACAAAAAIsIUgAAAABgEdufA0A7xJb5sIot8wHAGytSAAAAAGARQQoAAAAALCJIAQAAAIBFbSZILVu2TP369VOnTp0UFxen7du3+7skAAAAAG1UmwhSr7zyitLT0zV//nzt2rVLQ4cOVUJCgsrKyvxdGgAAAIA2qE3s2rdkyRLdc889uuuuuyRJmZmZeuutt/TCCy9o1qxZZ/Svrq5WdXW1+bmiokKS5PF4mqfgc2io/s7fJQAA4KWl/D8Srceg+W/7uwS0Ih8/kuDvEkyn/94ZhvGz/QKMc/Vo4WpqatS5c2e99tprGj9+vHk+JSVF5eXlevPNN8/4zsMPP6xHHnmkGasEAAAA0JocPXpUvXr1+sn2Vr8i9c9//lP19fVyOBxe5x0Ohz799NOzfmf27NlKT083Pzc0NOjEiRPq1q2bAgICzmu95+LxeNS7d28dPXpUNpvNr7WgZWKOoDGYJzgX5ggag3mCc2mLc8QwDJ08eVJRUVE/26/VBylfhISEKCQkxOtceHi4f4r5CTabrc1MRpwfzBE0BvME58IcQWMwT3AubW2O2O32c/Zp9ZtNdO/eXR06dFBpaanX+dLSUjmdTj9VBQAAAKAta/VBKjg4WLGxscrNzTXPNTQ0KDc3Vy6Xy4+VAQAAAGir2sRP+9LT05WSkqJhw4bpqquu0tNPP63KykpzF7/WJCQkRPPnzz/jp4fAacwRNAbzBOfCHEFjME9wLu15jrT6XftOe+655/TEE0/I7Xbrsssu09KlSxUXF+fvsgAAAAC0QW0mSAEAAABAc2n1z0gBAAAAQHMjSAEAAACARQQpAAAAALCIIAUAAAAAFhGkWpBly5apX79+6tSpk+Li4rR9+3Z/l4RmtGXLFt1www2KiopSQECA3njjDa92wzA0b9489ezZU6GhoYqPj9ehQ4e8+pw4cULJycmy2WwKDw9XamqqTp061YyjwPmUkZGhK6+8Ul27dlVkZKTGjx+vgwcPevWpqqpSWlqaunXrpi5duigpKemMF5YXFRUpMTFRnTt3VmRkpGbOnKm6urrmHArOkxUrVmjIkCGy2Wyy2WxyuVzauHGj2c78wNksWrRIAQEBmjZtmnmOudK+PfzwwwoICPA6BgwYYLYzP75HkGohXnnlFaWnp2v+/PnatWuXhg4dqoSEBJWVlfm7NDSTyspKDR06VMuWLTtr++LFi7V06VJlZmaqoKBAYWFhSkhIUFVVldknOTlZ+/fvV05OjrKzs7VlyxZNmjSpuYaA8ywvL09paWnatm2bcnJyVFtbq1GjRqmystLsM336dG3YsEHr1q1TXl6eiouLNWHCBLO9vr5eiYmJqqmp0datW7V69WplZWVp3rx5/hgSmlivXr20aNEiFRYWaufOnbruuus0btw47d+/XxLzA2fasWOHnn/+eQ0ZMsTrPHMFl156qUpKSszjgw8+MNuYH//HQItw1VVXGWlpaebn+vp6IyoqysjIyPBjVfAXScb69evNzw0NDYbT6TSeeOIJ81x5ebkREhJi/PWvfzUMwzAOHDhgSDJ27Nhh9tm4caMREBBgfP31181WO5pPWVmZIcnIy8szDOP7OdGxY0dj3bp1Zp9PPvnEkGTk5+cbhmEYf//7343AwEDD7XabfVasWGHYbDajurq6eQeAZnHBBRcY//3f/838wBlOnjxp9O/f38jJyTH+/d//3Zg6daphGPwtgWHMnz/fGDp06FnbmB//jxWpFqCmpkaFhYWKj483zwUGBio+Pl75+fl+rAwtxeHDh+V2u73miN1uV1xcnDlH8vPzFR4ermHDhpl94uPjFRgYqIKCgmavGedfRUWFJCkiIkKSVFhYqNraWq95MmDAAPXp08drngwePFgOh8Psk5CQII/HY65aoG2or6/X2rVrVVlZKZfLxfzAGdLS0pSYmOg1JyT+luB7hw4dUlRUlH71q18pOTlZRUVFkpgfPxTk7wIg/fOf/1R9fb3XZJMkh8OhTz/91E9VoSVxu92SdNY5crrN7XYrMjLSqz0oKEgRERFmH7QdDQ0NmjZtmq655hoNGjRI0vdzIDg4WOHh4V59fzxPzjaPTreh9du3b59cLpeqqqrUpUsXrV+/XjExMdqzZw/zA6a1a9dq165d2rFjxxlt/C1BXFycsrKydMkll6ikpESPPPKIrr32Wn388cfMjx8gSAFAK5SWlqaPP/7Y6zfrgCRdcskl2rNnjyoqKvTaa68pJSVFeXl5/i4LLcjRo0c1depU5eTkqFOnTv4uBy3QmDFjzH8PGTJEcXFx6tu3r1599VWFhob6sbKWhZ/2tQDdu3dXhw4dztjtpLS0VE6n009VoSU5PQ9+bo44nc4zNiepq6vTiRMnmEdtzJQpU5Sdna333ntPvXr1Ms87nU7V1NSovLzcq/+P58nZ5tHpNrR+wcHBuuiiixQbG6uMjAwNHTpUzzzzDPMDpsLCQpWVlemKK65QUFCQgoKClJeXp6VLlyooKEgOh4O5Ai/h4eG6+OKL9fnnn/O35AcIUi1AcHCwYmNjlZuba55raGhQbm6uXC6XHytDSxEdHS2n0+k1RzwejwoKCsw54nK5VF5ersLCQrPP5s2b1dDQoLi4uGavGU3PMAxNmTJF69ev1+bNmxUdHe3VHhsbq44dO3rNk4MHD6qoqMhrnuzbt88rdOfk5MhmsykmJqZ5BoJm1dDQoOrqauYHTCNHjtS+ffu0Z88e8xg2bJiSk5PNfzNX8EOnTp3SF198oZ49e/K35If8vdsFvrd27VojJCTEyMrKMg4cOGBMmjTJCA8P99rtBG3byZMnjd27dxu7d+82JBlLliwxdu/ebRw5csQwDMNYtGiRER4ebrz55pvG3r17jXHjxhnR0dHGv/71L/Mao0ePNi6//HKjoKDA+OCDD4z+/fsbt9xyi7+GhCY2efJkw263G++//75RUlJiHt99953Z59577zX69OljbN682di5c6fhcrkMl8tlttfV1RmDBg0yRo0aZezZs8fYtGmT0aNHD2P27Nn+GBKa2KxZs4y8vDzj8OHDxt69e41Zs2YZAQEBxjvvvGMYBvMDP+2Hu/YZBnOlvZsxY4bx/vvvG4cPHzY+/PBDIz4+3ujevbtRVlZmGAbz4zSCVAvy7LPPGn369DGCg4ONq666yti2bZu/S0Izeu+99wxJZxwpKSmGYXy/BfrcuXMNh8NhhISEGCNHjjQOHjzodY1vvvnGuOWWW4wuXboYNpvNuOuuu4yTJ0/6YTQ4H842PyQZq1atMvv861//Mu677z7jggsuMDp37mzceOONRklJidd1vvrqK2PMmDFGaGio0b17d2PGjBlGbW1tM48G58Pdd99t9O3b1wgODjZ69OhhjBw50gxRhsH8wE/7cZBirrRvN998s9GzZ08jODjY+Ld/+zfj5ptvNj7//HOznfnxvQDDMAz/rIUBAAAAQOvEM1IAAAAAYBFBCgAAAAAsIkgBAAAAgEUEKQAAAACwiCAFAAAAABYRpAAAAADAIoIUAAAAAFhEkAIAAAAAiwhSAAAAAGARQQoAAAAALCJIAQAAAIBF/wu4aQLjqXWm6AAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# The amount Fare didtribution\n",
    "titan[\"Fare\"].plot.hist(bins=5, figsize=(10,5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "bfff53fa-a73b-4cbb-a32c-5d01fc2e7e3d",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='SibSp', ylabel='count'>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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0a4JsQMYIDMzUwcPHuT+gu/o3LmziouLVVFRoffff18ZGRkqKCj40QfPyZMn9cwzzygvL0/Nmze3epyA8t2rFD169FBycrLat2+vd99990f/8ee7776rt99+W2vXrlW3bt1UXFysyZMnKy4uThkZGVaP94OInQDRunVrBQcHq7S01G99aWmpnE6nRVPhVjBx4kRt2rRJ27dvV7t27aweJ2CEhoaqY8eOkqTevXtr7969+t3vfqfXXnvN4smsVVRUpLKyMt1zzz2+dbW1tdq+fbteffVVVVVVKTg42MIJA0d0dLR+8pOf6Pjx41aPYrmpU6fqhRde0OjRoyVJ3bt313//938rJyfnlogd7tkJEKGhoerdu7fy8/N96+rq6pSfn899Brgqr9eriRMnav369frss8+UmJho9UgBra6uTlVVVVaPYbnBgwfrwIEDKi4u9i19+vRRenq6iouLCZ3vOH/+vL766iu1bdvW6lEsd+HCBQUF+SdDcHCw6urqLJro+nBlJ4BkZWUpIyNDffr0Ub9+/bR48WJVVlbqySeftHo0S50/f97vv6xOnDih4uJixcTEKCEhwcLJrJWZmam1a9fqgw8+UGRkpNxutyTJbrcrPDzc4umslZ2draFDhyohIUHffvut1q5dq23btumTTz6xejTLRUZGXnFfV8uWLdWqVasf/f1ezz33nIYPH6727dvr1KlTmjVrloKDg/X4449bPZrlhg8frnnz5ikhIUHdunXT559/rldeeUVjx461erRr40VAWbp0qTchIcEbGhrq7devn3fXrl1Wj2S5rVu3eiVdsWRkZFg9mqWu9p5I8q5atcrq0Sw3duxYb/v27b2hoaHeNm3aeAcPHuzdsmWL1WMFrPvuu8/7zDPPWD2G5UaNGuVt27atNzQ01Hv77bd7R40a5T1+/LjVYwUEj8fjfeaZZ7wJCQne5s2bezt06OD993//d29VVZXVo10Tfs8OAAAwGvfsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AC4ZdlsNm3YsEGS9PXXX8tms6m4uNjSmQAEHmIHQMA6c+aMJkyYoISEBIWFhcnpdCo1NVU7duyQJJ0+fVpDhw69rmOuX79e/fv3l91uV2RkpLp166bJkyc3wfQAAgV/CBRAwEpLS1N1dbXWrFmjDh06qLS0VPn5+Tp79qwkyel0Xtfx8vPzNWrUKM2bN0///M//LJvNpsOHDysvL68pxgcQIPjbWAACUnl5uW677TZt27ZN991331X3sdlsWr9+vUaMGKGvv/5aiYmJ+s///E8tWbJE+/fvV8eOHZWbm+t7/uTJk/XFF19o69at9b7u7NmztWHDBk2YMEFz587V2bNn9fDDD+uNN96Q3W5vknMF0LT4GAtAQIqIiFBERIQ2bNigqqqqa37e1KlT9eyzz+rzzz+Xy+XS8OHD/a4EHTp0SAcPHvzeYxw/flzvvvuuNm7cqM2bN+vzzz/XL3/5yxs6HwDWIXYABKSQkBCtXr1aa9asUXR0tAYMGKBf/epX+vOf//y9z5s4caLS0tLUtWtXLV++XHa7XStXrpQkTZo0SX379lX37t11xx13aPTo0XrzzTeviKmLFy/qrbfeUq9evTRw4EAtXbpU69atk9vtbrLzBdB0iB0AASstLU2nTp3Shx9+qAcffFDbtm3TPffco9WrV9f7HJfL5ft3SEiI+vTpoyNHjkiSWrZsqY8++kjHjx/X9OnTFRERoWeffVb9+vXThQsXfM9LSEjQ7bff7nfMuro6HT16tPFPEkCTI3YABLTmzZvrgQce0IwZM7Rz50794he/0KxZs27omHfeeaf+9V//Vf/xH/+h/fv36/Dhw3rnnXcaaWIAgYbYAXBLSUpKUmVlZb3bd+3a5fv3pUuXVFRUpK5du9a7/x133KEWLVr4HbOkpESnTp3yO2ZQUJA6d+58g9MDsAJfPQcQkM6ePatHH31UY8eOVY8ePRQZGal9+/ZpwYIFeuSRR+p9Xm5urjp16qSuXbtq0aJF+uabbzR27FhJ//im1YULF/TQQw+pffv2Ki8v15IlS1RTU6MHHnjAd4zmzZsrIyNDv/nNb+TxePT000/rscceu+6vugMIDMQOgIAUERGh5ORkLVq0SF999ZVqamoUHx+v8ePH61e/+lW9z5s/f77mz5+v4uJidezYUR9++KFat24tSbrvvvuUm5urMWPGqLS0VLfddpvuvvtubdmyxe+qTceOHTVy5Eg99NBDOnfunB5++GEtW7asyc8ZQNPg9+wAwHdc/j07/NkJwBzcswMAAIxG7AAAAKPxMRYAADAaV3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARvtfIzjyU5PysHwAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# The number of siblings and spouse accompanied with the passenger\n",
    "sns.countplot(x=\"SibSp\", data = titan)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "18508d54-837a-4f68-89fc-587b37999f82",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass\n",
       "3    491\n",
       "1    216\n",
       "2    184\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check for the number of classes\n",
    "titan['Pclass'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "ca43a473-dadc-4b6e-9fa9-82df5ddca15f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Pclass', ylabel='count'>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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lZTEqVUAgFn8bs4Ozk9iYqKWLVumq6++WsXFxWrQoIEuv/zyM/atrKzUvffeqzvvvFNt2rTRAw88oB07dnjnPwEAYALL78ZC7Rw8eFC9evXS22+/re3bt6uwsFCLFy/W9OnTdcstt6hPnz5yuVwaPHiwVq9ere+//16bNm3S448/rs8//1yS9Pjjj6usrEwzZ87UpEmTdOWVV+r++++3eGQAANQtzuxcpMLCwtStWze99NJLKigoUFVVleLj4zVq1Cg99thjstls+vjjj/X444/rvvvu895q3qNHD8XExCgnJ0czZszQunXrvLfev/XWW+rQoYPmzJmjMWPGWDxCAADqhs3j8XisLsJqbrdbDodDZWVlpz1z58SJEyosLFSLFi3UsGFDiyoE/x0AmIRbz39yvreen+v7++e4jAUAAIxG2AEAAEYj7AAAAKMRdgAAgNEIOwAAwGiEHQAAYDTCDgAAMBphBwAAGI2wYyiPx6PRo0crMjJSNptN+fn5ltTx/fffW/r+AADwcxG11Hniwnp9v21/Hvab+q9cuVLz589XTk6OWrZsqaZNm16gygAA8G+EHUMVFBQoNjZW1157rdWlAABgKS5jGWjEiBEaN26cioqKZLPZdPnll6umpkaZmZlq0aKFQkND1aFDBy1ZssS7T05Ojmw2m1atWqVOnTopNDRUvXr1UmlpqVasWKF27drJbrfrnnvu0bFjx7z7rVy5Utdff70iIiLUpEkT/f73v1dBQcE56/v666/Vv39/hYWFKSYmRkOHDtUPP/xwwf4eAIBLG2HHQC+//LKmTp2qZs2aaf/+/dq6dasyMzO1cOFCzZ07Vzt37tSECRN07733Kjc312ffp556Sn/961+1adMm7d27V3fccYdmzJihrKwsffTRR1q9erVmzZrl7V9eXq709HR9/vnnys7OVkBAgP7jP/5DNTU1Z6zt8OHD6tWrlzp16qTPP/9cK1euVElJie64444L+jcBAFy6uIxlIIfDofDwcAUGBsrpdKqiokLPPfec1q5dK5fLJUlq2bKlNmzYoFdffVU33nijd99nnnlG1113nSRp5MiRysjIUEFBgVq2bClJuu2227Ru3TpNmjRJkpSamurz3m+++aaioqK0a9cutW/f/rTa/vrXv6pTp0567rnnfPaJj4/X3//+d1155ZV1+8cAAFzyCDuXgD17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oD5x6/lP/PrW88LCwloXBgAAUJ9qNWcHAADgYlGrMzv333//OdvffPPNWhUDAABQ12oVdn788Uef9aqqKn399dc6fPjwGX8gFAAAwCq1CjtLly49bVtNTY3GjBmjK6644ryLAgAAqCt1NmcnICBA6enpeumll+rqkAAAAOetTicoFxQU6OTJk3V5SAAAgPNSq8tY6enpPusej0f79+/XRx99pOHDh9dJYQAAAHWhVmHnyy+/9FkPCAhQVFSU/vM///MX79QCAACoT7UKO+vWravrOgAAAC6IWoWdUw4cOKDdu3dLktq0aaOoqKg6KQoAAKCu1GqCcnl5ue6//37FxsaqR48e6tGjh+Li4jRy5EgdO3asrmsEAACotVqFnfT0dOXm5urDDz/U4cOHdfjwYX3wwQfKzc3Vn/70p7quEQAAoNZqdRnr/fff15IlS9SzZ0/vtgEDBig0NFR33HGH5syZU1f1AQAAnJdandk5duyYYmJiTtseHR3NZSwAAOBXahV2XC6XnnzySZ04ccK77fjx43r66aflcrnqrDgAAIDzVavLWDNmzFC/fv3UrFkzdejQQZL01VdfKSQkRKtXr67TAgEAAM5HrcJOUlKSvvvuO73zzjv69ttvJUl33323hgwZotDQ0DotEAAA4HzUKuxkZmYqJiZGo0aN8tn+5ptv6sCBA5o0aVKdFAcAAHC+ajVn59VXX1Xbtm1P237VVVdp7ty5510UAABAXalV2CkuLlZsbOxp26OiorR///7zLgoAAKCu1CrsxMfHa+PGjadt37hxo+Li4s67KAAAgLpSqzk7o0aN0vjx41VVVaVevXpJkrKzs/Xoo4/yBGUAAOBXahV2Jk6cqIMHD+qhhx5SZWWlJKlhw4aaNGmSMjIy6rRAAACA81GrsGOz2fTCCy/oiSee0DfffKPQ0FC1bt1aISEhdV0fAADAealV2DklLCxMXbt2rataAAAA6lytJigDAABcLAg7AADAaIQdAABgNMIOAAAwGmEHAAAYjbADAACMRtgBAABGszTsrF+/XoMGDVJcXJxsNpuWLVvm0z5ixAjZbDafpV+/fj59Dh06pCFDhshutysiIkIjR47U0aNH63EUAADAn1kadsrLy9WhQwfNnj37rH369eun/fv3e5f//u//9mkfMmSIdu7cqTVr1mj58uVav369Ro8efaFLBwAA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318JCQk6cuSIsrKylJOTo1WrVlldmvG49RznLScnRzfddNNp24cPH6758+fXf0G45NlstjNunzdvnkaMGFG/xQD/a+TIkcrOztb+/fvlcDiUnJysSZMm6eabb7a6NOMRdgAAgNF4gjIAADAaYQcAABiNsAMAAIxG2AEAAEYj7AAAAKMRdgAAgNEIOwAAwGiEHQAAYDTCDgCj9OzZU+PHj7e6DAB+hLADwO+MGDFCNptNNpvN+yvRU6dO1cmTJ60uDcBFiB8CBeCX+vXrp3nz5qmiokIff/yx0tLSFBQUpIyMDKtLA3CR4cwOAL8UEhIip9Op5s2ba8yYMerTp4/+9re/SZI2btyonj17qlGjRrrsssuUkpKiH3/88YzHeeutt9SlSxeFh4fL6XTqnnvuUWlpqbf9xx9/1JAhQxQVFaXQ0FC1bt1a8+bNkyRVVlZq7Nixio2NVcOGDdW8eXNlZmZe+MEDqFOc2QFwUQgNDdXBgweVn5+v3r176/7779fLL7+sBg0aaN26daqurj7jflVVVZo2bZratGmj0tJSpaena8SIEfr4448lSU888YR27dqlFStWqGnTptqzZ4+OHz8uSZo5c6b+9re/6b333lNCQoL27t2rvXv31tuYAdQNwg4Av+bxeJSdna1Vq1Zp3Lhxmj59urp06aJXXnnF2+eqq6466/7333+/93XLli01c+ZMde3aVUePHlVYWJiKiorUqVMndenSRZJ0+eWXe/sXFRWpdevWuv7662Wz2dS8efO6HyCAC47LWAD80vLlyxUWFqaGDRuqf//+uvPOO/XUU095z+z8Wtu2bdOgQYOUkJCg8PBw3XjjjZJ+CjKSNGbMGC1atEgdO3bUo48+qk2bNnn3HTFihPLz89WmTRs9/PDDWr16dd0OEkC9IOwA8Es33XST8vPz9d133+n48eNasGCBGjdurNDQ0F99jPLycqWkpMhut+udd97R1q1btXTpUkk/zceRpP79++uf//ynJkyYoH379ql379565JFHJElXX321CgsLNW3aNB0/flx33HGHbrvttrofLIALirADwC81btxYrVq1UkJCgho0+L8r7snJycrOzv5Vx/j222918OBBPf/887rhhhvUtm1bn8nJp0RFRWn48OF6++23NWPGDL322mveNrvdrjvvvFOvv/663n33Xb3//vs6dOjQ+Q8QQL1hzg6Ai0pGRoaSkpL00EMP6cEHH1RwcLDWrVun22+/XU2bNvXpm5CQoODgYM2aNUsPPvigvv76a02bNs2nz5QpU9S5c2ddddVVqqio0PLly9WuXTtJ0l/+8hfFxsaqU6dOCggI0OLFi+V0OhUREVFfwwVQBzizA+CicuWVV2r16tX66quvdM0118jlcumDDz7wOftzSlRUlObPn6/FixcrMTFRzz//vF588UWfPsHBwcrIyFBycrJ69OihwMBALVq0SJIUHh7unRDdtWtXff/99/r4448VEMD/dQIXE5vH4/FYXQQAAMCFwj9PAACA0Qg7AADAaIQdAABgNMIOAAAwGmEHAAAYjbADAACMRtgBAABGI+wAAACjEXYAAIDRCDsAAMBohB0AAGC0/w+aGv5BxK9FZgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Find the different P class columns\n",
    "sns.countplot(x='Pclass', data = titan, hue = 'Sex')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "e5313fc4-0c19-4b3b-8efe-e932e9ce464b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Pclass', ylabel='Age'>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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aBnMyAAA0EQNJ19L0uHDh2W1WhURESGr1eoKMT8dh+bFjAwAAE1UXl4uSfLz81NdXZ2qqqpUVVXl6m9obxiH5seMDAAATRQZGSlJqqurU7t27XTRRRcpNDRUF110kdq1a6e6ujq3cWh+zMgAANBE06dP19atWyVJf//73xUSEuLqq66u1s033+wah5bBjAwAAE2Un5/v+nrkyJHKycnR559/rpycHI0cObLRcWhezMgAANBEDTv7xsbGqrS0VJs2bdKmTZtc/Q3tp+8AjObV5BmZ2tpa7du3TydPnmzOegAAMI1u3bpJkmw2m9atW6fExETFxsYqMTFR69at08CBA93Gofmdd5A5fvy4Jk2apPbt2ys+Pl5lZWWSpGnTpunxxx9v9gIBAPBW9957rySpoKBA/v7+ysnJ0eLFi5WTkyN/f3+9/vrrbuPQ/M770lJGRob27Nmj9957T7/5zW9c7cnJyfrTn/6khx9+uFkLBADgXNTU1Lg+XHvS1VdfrT179iglJUXJyclKTEzU1q1b9e6778rhcOjqq6/WgQMHPF5XdHS064iE1uy8g8zq1au1cuVKDRw4UD4+Pq72+Ph4ffXVV81aHAAA56qsrEyTJ0827PUdDofefvttvf32227te/bsMaSuhQsXqkePHh5/XU877yDz73//W127dj2j/dixY27BBgAAT4qOjtbChQsNe/3a2lq98sor+vDDD3XNNddo4sSJ8vf3N6ye6Ohow17bk847yFx77bV66623NG3aNElyhZe//OUvstlszVsdAADnKDAw0PAZiPvuu0+TJ0/WfffdZ3gtbcV5B5nHHntMKSkp+vTTT3Xy5Ek999xz+vTTT/X+++9r8+bNLVEjAABAo877rqVBgwZp9+7dOnnypK666iq988476tq1q4qLi9W3b9+WqBEAAKBRTdpH5tJLL9XLL7+sDz74QJ9++qmWLl2qq6666oIKefzxx+Xj4+O2jXNNTY3S0tLUpUsXdezYUaNHj1ZlZeUFvQ4AAGg9zjvIVFdXN/o4evSoamtrm1TEjh079NJLLykhIcGtfcaMGVq7dq0KCgq0efNmlZeX69Zbb23SawAAgNbnvINMp06d1Llz5zMenTp1UlBQkGJiYpSVlSWn03lOz/fjjz8qNTVVL7/8sjp37uxqr6qq0qJFi/T0009r8ODB6tu3rxYvXqz3339f27ZtO9+yAQBAK3TeQWbJkiWKjIzUI488otWrV2v16tV65JFHdMkll2j+/PmaPHmynn/++XPe5TctLU033XSTkpOT3dp37dqluro6t/ZevXopOjpaxcXFZ30+u91+xmwRAABonc77rqVXX31VTz31lG677TZX24gRI3TVVVfppZdeUmFhoaKjo5WTk6NHHnnkvz7XihUr9OGHH2rHjh1n9FVUVMjf31+dOnVyaw8LC1NFRcVZn3PevHnKzs4+v18KAACY0nnPyLz//vvq06fPGe19+vRxzZQMGjToZ7eJPnDggH7/+98rPz+/WbdQzsjIUFVVlethxLbQAADAM847yERFRWnRokVntC9atEhRUVGSpB9++MFtvUtjdu3ape+++07XXHON2rVrp3bt2mnz5s16/vnn1a5dO4WFham2tlZHjhxx+7nKykqFh4ef9XkDAgIUEhLi9gAAAK3TeV9aevLJJzV27Fht2LBB/fr1kyTt3LlT//znP/XGG29IOnUX0rhx4/7r89xwww36+OOP3druvvtu9erVSw899JCioqLk5+enwsJCjR49WpK0b98+lZWVsYMwAACQ1IQgc/PNN2vfvn1asGCBPv/8c0lSSkqKVq9erR9//FGSNGXKlJ99nuDgYF155ZVubR06dFCXLl1c7ZMmTVJ6erpCQ0MVEhKiadOmyWazaeDAgedbNgAAaIXOO8hIUvfu3V13JVVXV2v58uUaN26cdu7cKYfD0WzFPfPMM/L19dXo0aNlt9s1dOhQ5eXlNdvzAwAAc2tSkJGkoqIiLVq0SG+88YYiIyN166236sUXX7ygYt577z237wMDA5Wbm6vc3NwLel4AANA6nVeQqaio0JIlS7Ro0SJVV1frtttuk91u1+rVq9W7d++WqhEAAKBR53zX0ogRI9SzZ0+VlJTo2WefVXl5uV544YWWrA0AAOC/OucZmQ0bNuh3v/udpkyZossvv7wlawIAADgn5zwjs2XLFh09elR9+/bVgAED9OKLL+r7779vydoAAAD+q3MOMgMHDtTLL7+sgwcP6t5779WKFSsUGRkpp9OpTZs26ejRoy1ZJwAAwBnOe2ffDh06aOLEidqyZYs+/vhjPfDAA3r88cfVtWtX3XzzzS1RIwAAQKPOO8icrmfPnnriiSf0zTffaPny5c1VEwAAwDm5oCDTwGKxaNSo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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# check for outliers for those passengers in pclass vs their ages(continous variable)\n",
    "sns.boxplot(x=\"Pclass\", y=\"Age\", data=titan)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "d11f6fdb-cb2a-4698-8324-c154b0b85e2f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Sex', ylabel='Age'>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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qqmpUU5OjsLCwgysFA3BFAviARcqJydH0reLJ44cOVLz589Xdna25s+fr5EjR6oup9f1A2AeCxculCSlpqbKarXqww8/VG5urj788ENZrVb9/Oc/9+mHxombRsLU6ua+9OzZUwUFBXrttdd8tte1M0cGMJ+6hSxbt26ttLQ0n/XDIiMjNXz4cJ9+aJwIMjC1urkvBQUF6tOnj9q2bavq6moFBwfr0KFDys/P9+kHwDzatWun3bt367nnnlPfvn316KOPKi4uTkVFRVq1apWef/55bz80XsyRgalxDh0IXMyBa9zO9vObERmYWt3KvTU1NRo+fLiuuOIKud1uBQUF6T//+Y9qamq8/bp162ZkqQDO0YEDByR9Owdu8ODB+vnPf67bbrtNmzdv1l/+8hfvHLgDBw5wfDdiBBmYWt3N4qKionT48GF99NFHPtvr2rmpHGA+dcdtnz59lJ+frzVr1mjNmjXe7b1799auXbs4vhs5ggxMre5mcYcPH5bValVcXJxCQ0NVVVWloqIiHT582KcfAPOoO25Hjx6tGTNmaOHChfriiy/Url073X///frss8+0a9cuju9GjiADU+vQoYP355YtW+rTTz/1Pr/88sv11VdfndYPgDl896aRs2bN0sSJE73buGkk6hBkYGpLlizx/lxbW6vhw4d7Tye9/fbbPv2++0cQaEyqqqq8N+M1m6FDh2rhwoWaOHGikpOT1bZtWx06dEg5OTnau3ev7r//fh08eNDoMs9LbGwsq9RfBAQZmNp315k4evSozzoyVqtVrVu31pdffsk6E2jUiouLNXbsWKPLuCCFhYUqLCw8rX3BggUGVHNxLFq0SB07djS6DNMjyMDUQkJCJElffvmlEhMT1atXL4WEhMjlcun9999XXl6eTz+gMYqNjdWiRYuMLuOCuN1ubd++XStXrtRdd92lfv36KSjI3IvTx8bGGl1CQCDIwNT69u2rHTt2yGKxaMaMGT7DtMnJyd51Jvr27WtglYCxQkNDA+Kbf1BQkFauXKmkpKSAeD+4OMwdZ9HoVVZWSvrfOhMLFy5USUmJFi5cqMGDB3vXmajrBwAILIzIwNTCw8O9/x47duy0dSbq2uv6AQACCyMyMLXLL79cknTs2LF6t9e11/UDAAQWggxMLSEhQVar9Qf7WK1W1pkAgADFqSWY2okTJ1RbWytJ6tmzp/r27eu9amnnzp0qKChQbW2tTpw4IbvdbnC1AICLjSADU8vIyJAkXXXVVSopKdFzzz3n3RYVFaUrr7xSBw8eVEZGhl588UWjygQANBCCDEytvLxckjRx4kRdffXVKiws1NGjRxUREaGEhATt27dPDz30kLcfACCwMEcGpuZwOCRJL7/8sqxWq7p166abb75Z3bp1k9Vq1SuvvOLTDwAQWAgyMLXZs2dLkt5///3T1oqprKxUQUGBTz8AQGDh1BJMzW63e28id9ttt6lnz54aPXq0XnnlFW+Iadu2LRN9ASBAEWRgeqtWrVJaWpoOHTqkgoICb4CRvg0xq1atMrA6AEBDIsggIKxatUpOp1MZGRkqLy+Xw+HQ7NmzGYkBgABHkEHAsNvtXGINAI0MQQaSpKqqKhUXFxtdBr4nNjbW547eAABfBBlIkoqLizV27Fijy8D3LFq0SB07djS6DADwWwQZSPr2m/+iRYuMLuOCff7555o9e7YyMjLUvn17o8u5YLGxsUaXAAB+jSADSVJoaGhAffNv3759QL0fAED9WBAPAACYFkEGAACYlqFB5vHHH5fFYvF5dO7c2bu9qqpK6enpatWqlZo3b66UlBRu/gcAALwMH5G59tprdfjwYe9j+/bt3m2TJk3Sxo0btW7dOm3dulWlpaUaNmyYgdUCAAB/Yvhk3yZNmigyMvK0dqfTqaVLl2r16tXq37+/JGnZsmW6+uqrlZ+frz59+lzqUgEAgJ8xfETmwIEDio6OVnx8vNLS0ryLsu3Zs0c1NTUaMGCAt2/nzp0VGxurvLy8M+7P5XKpoqLC5wEAAAKToUGmd+/eWr58ubZs2aKXXnpJRUVF+ulPf6rjx4+rrKxMwcHBCg8P93mNw+FQWVnZGfeZmZkpu93ufcTExDTwuwAAAEYx9NRScnKy9+eEhAT17t1b7du312uvvaawsLDz2ufUqVM1efJk7/OKigrCDAAAAcrwU0vfFR4ero4dO+rTTz9VZGSkqqurdezYMZ8+5eXl9c6pqRMSEiKbzebzAAAAgcmvgsyJEyd08OBBRUVFqUePHmratKlyc3O92/fv36/i4mIlJiYaWCUAAPAXhp5amjJligYPHqz27durtLRUM2bMkNVq1ahRo2S32zVmzBhNnjxZERERstlsmjBhghITE7liCQAASDI4yHzxxRcaNWqUvv76a7Vu3Vr9+vVTfn6+WrduLUl69tlnFRQUpJSUFLlcLg0cOFDz5883smQAAOBHDA0ya9eu/cHtoaGhysrKUlZW1iWqCAAAmIlfzZEBAAA4FwQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWgQZAABgWucdZKqrq7V//3598803F7MeAACAs3bOQaayslJjxozRZZddpmuvvVbFxcWSpAkTJmjOnDkXvUAAAIAzOecgM3XqVH388cd67733FBoa6m0fMGCAXn311YtaHAAAwA9pcq4vyM7O1quvvqo+ffrIYrF426+99lodPHjwohYHAADwQ855RObLL79UmzZtTms/efKkT7ABAABoaOccZG644QZt2rTJ+7wuvCxZskSJiYkXrzIAAIAfcc6nlp566iklJyfrk08+0TfffKPnnntOn3zyiXbu3KmtW7c2RI0AAAD1OucRmX79+umjjz7SN998o65du+qtt95SmzZtlJeXpx49ejREjQAAAPU6r3VkrrzySi1evFjvv/++PvnkE61cuVJdu3a9oELmzJkji8WiiRMnetuqqqqUnp6uVq1aqXnz5kpJSVF5efkF/R4AABA4zjnIVFRU1Ps4fvy4qqurz6uIgoICLVy4UAkJCT7tkyZN0saNG7Vu3Tpt3bpVpaWlGjZs2Hn9DgAAEHjOOciEh4erZcuWpz3Cw8MVFham9u3ba8aMGXK73We1vxMnTigtLU2LFy9Wy5Ytve1Op1NLly7Vn/70J/Xv3189evTQsmXLtHPnTuXn559r2QAAIACdc5BZvny5oqOjNW3aNGVnZys7O1vTpk1T27Zt9dJLL2ns2LF6/vnnz3qV3/T0dN1+++0aMGCAT/uePXtUU1Pj0965c2fFxsYqLy/vjPtzuVynjRYBAIDAdM5XLa1YsULPPPOMhg8f7m0bPHiwunbtqoULFyo3N1exsbGaPXu2pk2b9oP7Wrt2rT744AMVFBSctq2srEzBwcEKDw/3aXc4HCorKzvjPjMzMzVz5sxze1MAAMCUznlEZufOnerWrdtp7d26dfOOlPTr1897D6YzKSkp0W9+8xutWrXK51YHF2rq1KlyOp3eR0lJyUXbNwAA8C/nHGRiYmK0dOnS09qXLl2qmJgYSdLXX3/tM9+lPnv27NGRI0fUvXt3NWnSRE2aNNHWrVv1/PPPq0mTJnI4HKqurtaxY8d8XldeXq7IyMgz7jckJEQ2m83nAQAAAtM5n1r64x//qNTUVOXk5Khnz56SpN27d+uf//yn1q9fL+nbq5BGjBjxg/u5+eabtXfvXp+2e++9V507d9bvf/97xcTEqGnTpsrNzVVKSookaf/+/SouLmYFYQAAIOk8gswdd9yh/fv3a8GCBfr3v/8tSUpOTlZ2drZOnDghSRo3btyP7qdFixbq0qWLT1uzZs3UqlUrb/uYMWM0efJkRUREyGazacKECUpMTFSfPn3OtWwAABCAzjnISNIVV1zhvSqpoqJCa9as0YgRI7R7927V1tZetOKeffZZBQUFKSUlRS6XSwMHDtT8+fMv2v4BAIC5nVeQkaRt27Zp6dKlWr9+vaKjozVs2DC9+OKLF1TMe++95/M8NDRUWVlZysrKuqD9AgCAwHROQaasrEzLly/X0qVLVVFRoeHDh8vlcik7O1vXXHNNQ9UIAABQr7O+amnw4MHq1KmTCgsLNW/ePJWWluqFF15oyNoAAAB+0FmPyOTk5Oihhx7SuHHj1KFDh4asCQAA4Kyc9YjM9u3bdfz4cfXo0UO9e/fWiy++qK+++qohawMAAPhBZx1k+vTpo8WLF+vw4cO6//77tXbtWkVHR8vtduvtt9/W8ePHG7JOAACA05zzyr7NmjXTfffdp+3bt2vv3r367W9/qzlz5qhNmza64447GqJGAACAep1zkPmuTp066emnn9YXX3yhNWvWXKyaAAAAzsoFBZk6VqtVQ4cO1RtvvHExdgcAAHBWLkqQAQAAMAJBBgAAmBZBBgAAmBZBBgAAmBZBBgAAmNZ53/0avsrLy+V0Oo0uo9H7/PPPff6Fsex2uxwOh9FlXDCOb//A8e1f/OX4tng8Ho/RRTSkiooK2e12OZ1O2Wy2Bvkd5eXlumv0L1VT7WqQ/QNm1TQ4RCtfedkv/tidL45voH4NfXyf7ec3IzIXgdPpVE21S6fib5Q71G50OYBfCKpySp9tldPpNHWQ4fgGTudPxzdB5iJyh9rlbna50WUAaAAc34B/YrIvAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLYIMAAAwLUODzEsvvaSEhATZbDbZbDYlJiYqJyfHu72qqkrp6elq1aqVmjdvrpSUFJWXlxtYMQAA8CeGBpl27dppzpw52rNnj3bv3q3+/ftryJAh2rdvnyRp0qRJ2rhxo9atW6etW7eqtLRUw4YNM7JkAADgR5oY+csHDx7s83z27Nl66aWXlJ+fr3bt2mnp0qVavXq1+vfvL0latmyZrr76auXn56tPnz5GlAwAAPyI38yRqa2t1dq1a3Xy5EklJiZqz549qqmp0YABA7x9OnfurNjYWOXl5Z1xPy6XSxUVFT4PAAAQmAwPMnv37lXz5s0VEhKiBx54QK+//rquueYalZWVKTg4WOHh4T79HQ6HysrKzri/zMxM2e127yMmJqaB3wEAADCK4UGmU6dO+uijj7Rr1y6NGzdOd999tz755JPz3t/UqVPldDq9j5KSkotYLQAA8CeGzpGRpODgYF111VWSpB49eqigoEDPPfecRowYoerqah07dsxnVKa8vFyRkZFn3F9ISIhCQkIaumwAAOAHDA8y3+d2u+VyudSjRw81bdpUubm5SklJkSTt379fxcXFSkxMNLjK+gWdOmZ0CYDfCLTjIdDeD3Ah/Ol4MDTITJ06VcnJyYqNjdXx48e1evVqvffee3rzzTdlt9s1ZswYTZ48WREREbLZbJowYYISExP99oqlsKJtRpcAoIFwfAP+ydAgc+TIEf3yl7/U4cOHZbfblZCQoDfffFO33HKLJOnZZ59VUFCQUlJS5HK5NHDgQM2fP9/Ikn/QqbgkucPCjS4D8AtBp44F1Ic/xzfwP/50fBsaZJYuXfqD20NDQ5WVlaWsrKxLVNGFcYeFy93scqPLANAAOL4B/2T4VUsAAADniyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMiyADAABMq4nRBQSSoCqn0SUAfoPjAcClQJC5COx2u5oGh0ifbTW6FMCvNA0Okd1uN7oMAAGMIHMR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a8uWLadNAAYAAI2P368jc6FYRwYAAPM5289vv57sCwAA8EMIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLT8fkG8C1W3TE5FRYXBlQAAgLNV97n9Y8vdBXyQOX78uCQpJibG4EoAAMC5On78uOx2+xm3B/zKvm63W6WlpWrRooUsFovR5aCBVVRUKCYmRiUlJazkDAQYju/GxePx6Pjx44qOjlZQ0JlnwgT8iExQUJDatWtndBm4xGw2G3/ogADF8d14/NBITB0m+wIAANMiyAAAANMiyCCghISEaMaMGQoJCTG6FAAXGcc36hPwk30BAEDgYkQGAACYFkEGAACYFkEGAACYFkEGjcI999yjoUOHGl0G0Ch4PB6NHTtWERERslgs+uijjwyp4z//+Y+hvx+XRsAviAcAuLS2bNmi5cuX67333lN8fLwuv/xyo0tCACPIAAAuqoMHDyoqKkp9+/Y1uhQ0Apxagt+56aabNGHCBE2cOFEtW7aUw+HQ4sWLdfLkSd17771q0aKFrrrqKuXk5EiSamtrNWbMGMXFxSksLEydOnXSc88994O/w+12KzMz0/ua6667Tn/5y18uxdsDAto999yjCRMmqLi4WBaLRVdcccWPHm/vvfeeLBaL3nzzTXXr1k1hYWHq37+/jhw5opycHF199dWy2Wz6xS9+ocrKSu/rtmzZon79+ik8PFytWrXSoEGDdPDgwR+s7x//+IeSk5PVvHlzORwOjR49Wl999VWD/fdAwyPIwC+tWLFCl19+ud5//31NmDBB48aNU2pqqvr27asPPvhAt956q0aPHq3Kykq53W61a9dO69at0yeffKLHHntM06ZN02uvvXbG/WdmZurll1/WggULtG/fPk2aNEl33XWXtm7degnfJRB4nnvuOT3xxBNq166dDh8+rIKCgrM+3h5//HG9+OKL2rlzp0pKSjR8+HDNmzdPq1ev1qZNm/TWW2/phRde8PY/efKkJk+erN27dys3N1dBQUG688475Xa7663t2LFj6t+/v7p166bdu3dry5YtKi8v1/Dhwxv0vwkamAfwMzfeeKOnX79+3ufffPONp1mzZp7Ro0d72w4fPuyR5MnLy6t3H+np6Z6UlBTv87vvvtszZMgQj8fj8VRVVXkuu+wyz86dO31eM2bMGM+oUaMu4jsBGqdnn33W0759e4/Hc3bH27vvvuuR5HnnnXe82zMzMz2SPAcPHvS23X///Z6BAwee8fd++eWXHkmevXv3ejwej6eoqMgjyfPhhx96PB6P58knn/TceuutPq8pKSnxSPLs37//vN8vjMUcGfilhIQE789Wq1WtWrVS165dvW0Oh0OSdOTIEUlSVlaW/vznP6u4uFinTp1SdXW1rr/++nr3/emnn6qyslK33HKLT3t1dbW6det2kd8J0Lidy/H23ePe4XDosssuU3x8vE/b+++/731+4MABPfbYY9q1a5e++uor70hMcXGxunTpclotH3/8sd599101b978tG0HDx5Ux44dz+9NwlAEGfilpk2b+jy3WCw+bRaLRdK3c13Wrl2rKVOm6JlnnlFiYqJatGihP/zhD9q1a1e9+z5x4oQkadOmTWrbtq3PNu7hAlxc53K8ff8Yr+/vwHdPGw0ePFjt27fX4sWLFR0dLbfbrS5duqi6uvqMtQwePFhz5849bVtUVNS5vTH4DYIMTG/Hjh3q27evxo8f7237oQl/11xzjUJCQlRcXKwbb7zxUpQINFoNdbx9/fXX2r9/vxYvXqyf/vSnkqTt27f/4Gu6d++u9evX64orrlCTJnz8BQr+l4TpdejQQS+//LLefPNNxcXF6ZVXXlFBQYHi4uLq7d+iRQtNmTJFkyZNktvtVr9+/eR0OrVjxw7ZbDbdfffdl/gdAIGroY63li1bqlWrVlq0aJGioqJUXFysRx555Adfk56ersWLF2vUqFF6+OGHFRERoU8//VRr167VkiVLZLVaz6sWGIsgA9O7//779eGHH2rEiBGyWCwaNWqUxo8f7708uz5PPvmkWrdurczMTH322WcKDw9X9+7dNW3atEtYOdA4NMTxFhQUpLVr1+qhhx5Sly5d1KlTJz3//PO66aabzvia6Oho7dixQ7///e916623yuVyqX379vq///s/BQVxEa9ZWTwej8foIgAAAM4HERQAAJgWQQYAAJgWQQYAAJgWQQYAAJgWQQYAAJgWQQYAAJgWQQYAAJgWQQYAAJgWQQYAAJgWQQaA3/nyyy81btw4xcbGKiQkRJGRkRo4cKB27NhhdGkA/Az3WgLgd1JSUlRdXa0VK1YoPj5e5eXlys3N1ddff210aQD8DCMyAPzKsWPH9Pe//11z587Vz372M7Vv3169evXS1KlTdccdd3j7/OpXv1Lr1q1ls9nUv39/ffzxx5K+Hc2JjIzUU0895d3nzp07FRwcrNzcXEPeE4CGQ5AB4FeaN2+u5s2bKzs7Wy6Xq94+qampOnLkiHJycrRnzx51795dN998s44eParWrVvrz3/+sx5//HHt3r1bx48f1+jRo/Xggw/q5ptvvsTvBkBD4+7XAPzO+vXr9etf/1qnTp1S9+7ddeONN2rkyJFKSEjQ9u3bdfvtt+vIkSMKCQnxvuaqq67Sww8/rLFjx0qS0tPT9c477+iGG27Q3r17VVBQ4NMfQGAgyADwS1VVVfr73/+u/Px85eTk6P3339eSJUt08uRJPfTQQwoLC/Ppf+rUKU2ZMkVz5871Pu/SpYtKSkq0Z88ede3a1Yi3AaCBEWQAmMKvfvUrvf322xo/frxeeOEFvffee6f1CQ8P1+WXXy5J+sc//qGePXuqpqZGr7/+ugYPHnyJKwZwKXDVEgBTuOaaa5Sdna3u3burrKxMTZo00RVXXFFv3+rqat11110aMWKEOnXqpF/96lfau3ev2rRpc2mLBtDgGJEB4Fe+/vprpaam6r777lNCQoJatGih3bt3a8KECbr99tu1ZMkSJSUl6fjx43r66afVsWNHlZaWatOmTbrzzjt1ww036He/+53+8pe/6OOPP1bz5s114403ym63629/+5vRbw/ARUaQAeBXXC6XHn/8cb311ls6ePCgampqFBMTo9TUVE2bNk1hYWE6fvy4MjIytH79eu/l1klJScrMzNTBgwd1yy236N1331W/fv0kSf/5z3903XXXac6cORo3bpzB7xDAxUSQAQAApsU6MgAAwLQIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLQIMgAAwLT+P69LTwgG+aIrAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# check for outliers for Gender vs their ages(continous variable)\n",
    "sns.boxplot(x=\"Sex\", y=\"Age\", data=titan)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e1ead1b-6254-49c1-a0d3-a5021976b660",
   "metadata": {},
   "source": [
    "# Bivaraite Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "bd202c9f-51e8-4322-8f11-f1ff0feac17c",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Sex', ylabel='Survived'>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AQAARiHcAAAAoxBuAACAUQg3AADAKF1+FPxnP/tZl3f66KOP9qgYAACAY9XlmZvvv3wyJiZGbrdbu3btCm6vqamR2+1WbGxsnxQKAADQFV2eufmP//iP4N833HCDLrvsMpWWlio8PFzS316DsGzZMn47BgAAWKpH99yUlZXpuuuuCwYbSQoPD5fT6VRZWVmvFQcAANBdPQo333zzjd55553D2t955x35/f5jLgoAAKCnevRuqfz8fC1evFh79+5Venq6JOmll17S+vXrgy+8BAAAsEKPws3tt9+u+Ph43XHHHdq3b58kacyYMVq1apX++Z//uVcLBAAA6I4ehZuwsDBdf/31uv766+Xz+SSJG4kBAEBI6PGP+H3zzTd65pln9Kc//Uk2m02S9Omnn+rAgQO9VhwAAEB39Wjm5qOPPtKFF16ouro6tba2atasWRo2bJg2bNig1tZWlZaW9nadAAAAXdKjmZtrrrlGaWlp+uKLLzRkyJBg+8UXXyy3291rxQEAAHRXj2Zunn/+ee3cuVMREREd2pOSkvTJJ5/0SmEAAAA90aOZG7/fr/b29sPaP/74Yw0bNuyYiwIAAOipHoWb2bNnq7i4OLhus9l04MABFRUVac6cOb1VGwAAQLf16LLUHXfcoezsbCUnJ+vQoUO64oor9N577ykuLk5/+tOfertGAACALutRuDnppJP02muvqby8XK+//roOHDigxYsX6+c//3mHG4wBAAD6W4/CzaFDhxQZGakrr7yyt+sBAAA4Jj2652b06NHKy8vT008/zYsyAQBASOlRuLn//vt18OBBXXTRRUpISNCvf/1r7dq1q7drAwAA6LYehZuLL75YDz30kLxer2677Ta99dZbOvvsszV+/HitXbu2t2sEAADosh6/W0qShg0bpvz8fG3fvl2vv/66hg4dqltvvbW3agMAAOi2Ywo3hw4d0oMPPqj58+drypQp+vzzz7Vq1areqg0AAKDbevS01FNPPaWtW7fqscce06BBg3TppZdq+/btmj59em/XBwAA0C09CjcXX3yxfvKTn2jLli2aM2eOBg8e3Nt1AQAA9EiPwo3X6+UdUgAAICR1Odz4fD7FxMRIkgKBgHw+3xH7ftsPAACgv3U53IwYMUL79u3T6NGjNXz4cNlstsP6BAIB2Wy2Tt8YDgAA0B+6HG7+93//VyNHjgz+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k0+qurpa9957r1paWvSrX/1KQ4YM6dD/q6++0nXXXacNGzYE1ydMmKD6+nrV1NTorLPOsuI0APQxwg2AAWvJkiV6+umntWzZMt11113asWPHYX2GDx+uuLg4SdIbb7yhqVOn6uuvv9Z//dd/ad68ef1cMYD+wNNSAAas5ORkPfbYY5oyZYo8Ho8GDRqkpKSkTvu2tbXpyiuvVE5Ojk4//XQtWbJEu3fv1ujRo/u3aAB9jpkbACHvs88+04IFC/SP//iPmjhxooYNG6Zdu3Zp5cqVmjt3ru69915Nnz5d+/fv129+8xuNHz9en376qbZt26aLL75YaWlpWrVqlR5++GG99tprio6O1owZMxQbG6s///nPVp8egF5GuAEQ8lpbW7VmzRpt375de/fu1ddff63ExEQtWLBAN954o4YMGaL9+/frpptu0iOPPBJ89Hv69OlyuVzau3evZs2apWeffVbnnnuuJOnDDz/UpEmTtH79el199dUWnyGA3kS4AQAARuF3bgAAgFEINwAAwCiEGwAAYBTCDQAAMArhBgAAGIVwAwAAjEK4AQAARiHcAAAAoxBuAACAUQg3AADAKIQbAABglP8HxHmmyIYn/zUAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#which gender has better chances of survival\n",
    "sns.barplot(x='Sex', y = 'Survived', data = titan)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "55e3ba2b-4ed8-4de8-bffe-a4b3bf7ba740",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Pclass', ylabel='Survived'>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# which passenger class has a better chance of survival\n",
    "sns.barplot(x = 'Pclass', y = 'Survived', data = titan)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dbffdadd-4529-4ebf-82ef-655858c44c37",
   "metadata": {},
   "source": [
    "# Feature Enconding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "1d6c90a5-feb1-439b-9ce7-b6dfe5280f7a",
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Embarked  Gender  \n",
       "0      0         A/5 21171   7.2500        S       1  \n",
       "1      0          PC 17599  71.2833        C       0  \n",
       "2      0  STON/O2. 3101282   7.9250        S       0  \n",
       "3      0            113803  53.1000        S       0  \n",
       "4      0            373450   8.0500        S       1  "
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titan['Gender'] = titan['Sex'].map({'male':1, 'female':0})\n",
    "titan.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "0b89f82f-cda7-4c29-bb6c-53de3fe7fac0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop Categorical variables not required \n",
    "titan.drop(['Name','Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "984cd86b-01be-4a30-8cdf-b633b6df4c23",
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass   Age  SibSp  Parch     Fare  Gender\n",
       "0            1         0       3  22.0      1      0   7.2500       1\n",
       "1            2         1       1  38.0      1      0  71.2833       0\n",
       "2            3         1       3  26.0      0      0   7.9250       0\n",
       "3            4         1       1  35.0      1      0  53.1000       0\n",
       "4            5         0       3  35.0      0      0   8.0500       1"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titan.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "8a4813c9-b20d-4cd2-b49e-70a4a36dd646",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Seperate Dependent and Independent variables\n",
    "x = titan[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]\n",
    "y = titan['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "116ca47d-357f-4287-9820-0afa231ccc53",
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
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>3</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass   Age  SibSp  Parch     Fare  Gender\n",
       "0            1       3  22.0      1      0   7.2500       1\n",
       "1            2       1  38.0      1      0  71.2833       0\n",
       "2            3       3  26.0      0      0   7.9250       0\n",
       "3            4       1  35.0      1      0  53.1000       0\n",
       "4            5       3  35.0      0      0   8.0500       1"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "cd68c51b-7276-4e2c-bd15-3aadc2907633",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0\n",
       "1    1\n",
       "2    1\n",
       "3    1\n",
       "4    0\n",
       "Name: Survived, dtype: int64"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c35a0c76-6d8f-4a72-86fb-4de184b86606",
   "metadata": {},
   "source": [
    "# Build the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "fa0e6d19-0464-4dc2-bf2c-b7f171cbfb4b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(891, 7) (623, 7) (268, 7)\n"
     ]
    }
   ],
   "source": [
    "# Train test split the data\n",
    "from sklearn.model_selection import train_test_split\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)\n",
    "print(x.shape, x_train.shape, x_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "111fbbf0-aa5e-4c0b-9528-951f2babf6aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import Decision Tree regression\n",
    "from sklearn.tree import DecisionTreeClassifier "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "9aba0e79-9aff-4cd1-a009-fee28c047c3e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-1 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-1 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-1 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-1 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-1 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;DecisionTreeClassifier<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeClassifier.html\">?<span>Documentation for DecisionTreeClassifier</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>DecisionTreeClassifier()</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fit Decision Tree into the model\n",
    "dt = DecisionTreeClassifier()\n",
    "dt.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8899727-3498-4e21-981a-1f76bf42f8ce",
   "metadata": {},
   "source": [
    "# Predicting Our Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "2bacd9ff-d7f5-40f3-9639-c40e5d608224",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Predict our model\n",
    "dt = DecisionTreeClassifier()\n",
    "predictions = dt.fit(x_train,y_train)\n",
    "predictions = dt.predict([[1,3,22,1,0,7.2500,1],[2,1,38.0,1,0,71.2833,0]])\n",
    "predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "a0d5baec-c4ce-4a3e-a326-e9233672e101",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1,\n",
       "       0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1,\n",
       "       0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,\n",
       "       0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,\n",
       "       1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0,\n",
       "       0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,\n",
       "       0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,\n",
       "       0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0,\n",
       "       1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0,\n",
       "       0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0,\n",
       "       1, 0, 0, 1])"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Calculating accuracy of our model\n",
    "dt = DecisionTreeClassifier()\n",
    "dt.fit(x_train,y_train)\n",
    "predictions = dt.predict(x_test)\n",
    "predictions"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4962ef80-8ef7-4f50-9731-dd76300ff248",
   "metadata": {},
   "source": [
    "# Testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "396c5c48-4364-4559-8e23-dfbde8e5703c",
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
       "      <th>Predicted No</th>\n",
       "      <th>Predicted Yes</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Actual No</th>\n",
       "      <td>130</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Actual Yes</th>\n",
       "      <td>32</td>\n",
       "      <td>79</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Predicted No  Predicted Yes\n",
       "Actual No            130             27\n",
       "Actual Yes            32             79"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Print Confussion Matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "cm = confusion_matrix(y_test, predictions)\n",
    "cm = pd.DataFrame(confusion_matrix(y_test, predictions),columns = ['Predicted No', 'Predicted Yes'], index=['Actual No', 'Actual Yes'])\n",
    "cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "ae814987-286d-4deb-8aed-1db1927a1cd0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.80      0.83      0.82       157\n",
      "           1       0.75      0.71      0.73       111\n",
      "\n",
      "    accuracy                           0.78       268\n",
      "   macro avg       0.77      0.77      0.77       268\n",
      "weighted avg       0.78      0.78      0.78       268\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# View classification report\n",
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "9cd79fc1-728a-49e2-a190-432ca6e83625",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7798507462686567"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Analyze accuracy of our model\n",
    "from sklearn.metrics import accuracy_score\n",
    "result = accuracy_score(y_test, predictions)\n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec33ae80-6f75-4cf8-be98-f14103f76018",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
 Predicting Survival Rate(Titanic).ipynb…]()




# [Project 3: Power BI HR Attrition Dashboard] 

### Project Goal: Is to analyze the HR Attrition currently taking place in the organization

### I collected the data, stored, transformed, and retrieved it for analysis and reporting purposes. Using query editor, I was able to promote the headers as this was missing from the original dataset.  I also transformed the Age column to integer, changed salary type from monthly to yearly and added the salary column to the dataset. I also added a conditional column called Attrition count.
### In the data model view, I computed the Age bins category to make our analysis much more efficient. I also calculated measures like Attrition rate, Average salary, Max salary, Min salary and Active employees.
### I created reports using Card, column charts, clustered bar chart, donut chart, pie chart, line chart and slicers for comparison and analysis purposes. The reports were shared within user groups using power Bi apps and then published to power Bi services. By pinning the visuals from the reports, I created a dashboard. I implemented static and dynamic Row level security in the data model in power bi desktop and assigned roles to the users.
## Overview of the Dashboard 
![image](https://github.com/sirskin01/try01/assets/144762826/2d3bdf81-4560-43a0-b4e8-e7c3381de747)

# [Project 4: Tableau Sample SuperStore Dashboard] 

### Project Goal: Is to analyze the company's sales and profit performances. The dashboard also shows the different kPI’s as well as the current year to date sales of all categories and subcategories. 
### I imported the data from Kaggle, checked for null values. I created the different KPI’s, which include sales kpi, profit kpi and quantity kpi to show the amount of progress made towards the different measurable goals. The YTD vs PYTD of sub-categories shows the phone categories are performing better than the labels. 
![image](https://github.com/sirskin01/try01/assets/144762826/2eea9331-85a1-4261-b54e-4ed490ed2485)

## Weekly Dashboard: 
### This dashboard was created to track the weeekly performance of the store in terms of sales and profits
![image](https://github.com/sirskin01/try01/assets/144762826/c3339b6c-6060-4c7b-a05c-758fc93a48c8)

# [Project 6: Knime Markerting Analysis]
