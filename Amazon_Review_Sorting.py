import pandas as pd
import datetime as dt
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/amazon_review.csv")

# Data Preprocessing
df.shape  # There are 4915 comments.
df.dtypes

df["asin"].nunique()  # There is 1 product.
df["reviewerID"].nunique() # 4915 unique users commented.
df.isnull().sum() # Missing value in Comment and Commenters name.
df[df["reviewText"].isna()] # The user who did not write anything in the comment line gave 5 ratings and 2 out of 3 people
# found it useful, I am not deleting this comment.
df.drop("helpful", axis=1, inplace=True)   # I'm deleting the helpful variable as it has helpful_yes and total_vote variables.


# Average rating
df["overall"].mean()

# Time-based Weighted Average
time_weighted_mean = df.loc[df["day_diff"] <= 30, "overall"].mean() * 28 / 100 + \
                     df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 26 / 100 + \
                     df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 24 / 100 + \
                     df.loc[(df["day_diff"] > 180), "overall"].mean() * 22 / 100
time_weighted_mean
df["overall"].mean()
diff = time_weighted_mean - df["overall"].mean()

(time_weighted_mean - df["overall"].mean()) / df["overall"].mean() * 100
# There is a difference of 2.422341944993339%. Taking the average, taking into account the current reviews, increased the
# product rating by 2.42%.


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

     - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
     - The score to be calculated is used for product ranking.

    - Note:
     If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be made to conform to Bernoulli.
     This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["WLB_Score"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("WLB_Score", ascending=False).loc[:,["reviewerName","summary", "overall"]].head(20)
# Top 20 reviews that accurately reflect the product based on the Wilson Lower Bound Score